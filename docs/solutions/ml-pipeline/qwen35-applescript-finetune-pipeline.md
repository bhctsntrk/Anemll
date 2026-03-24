---
title: "End-to-end: Fine-tune Qwen3.5-0.8B for AppleScript Generation on ANE"
category: ml-pipeline
tags:
  - qwen3.5
  - deltanet
  - applescript
  - apple-neural-engine
  - lora
  - sft
  - coreml
  - macos-automation
  - runpod
module: spotlight-ai
symptom: "Qwen3.5-0.8B has no AppleScript knowledge — produces generic text instead of valid macOS automation scripts"
root_cause: "Base model not trained on AppleScript domain; requires SFT with curated instruction→code pairs and DeltaNet-aware LoRA configuration"
severity: project
date: 2026-03-24
---

# Fine-tune Qwen3.5-0.8B for AppleScript Generation on ANE

## Problem

Qwen3.5-0.8B runs on Apple Neural Engine at ~20 t/s via ANEMLL but has zero AppleScript knowledge. Given "Finder'da dosyaları listele", it produces generic Turkish text instead of `tell application "Finder" to get name of every file of desktop`.

## Solution: 5-Phase Pipeline

### Phase 1: Data Collection (5261 verified pairs, 57 apps)

**6 data sources, all free:**

| Source | Raw | After osacompile verify |
|--------|-----|------------------------|
| sdef parsing (18 app dictionaries) | — | Structured command/class/property data |
| macos-automator-mcp (steipete/macos-automator-mcp) | 212 | ~88 |
| GitHub repos (3 collections) | 122 | ~122 |
| macOS .scpt decompiles (`osadecompile`) | 63 | ~51 |
| Template expansion (EN+TR) | 334 | 334 |
| Synthetic generation (Gemini Flash Lite via OpenRouter) | 4450 | ~4100 |
| **Total** | **6885** | **5261** |

**Key discoveries:**
- `sdef` CLI requires Xcode, but `.sdef` XML files are readable directly from `App.app/Contents/Resources/*.sdef`
- `osacompile` validates class/verb names but NOT property names — `get bogusproperty of front window` compiles cleanly
- `osadecompile` can recover AppleScript source from compiled `.scpt` files (71 files from `/Library/Scripts/`, 80 from `/System/Library/Templates/`)
- Synthetic data via Gemini Flash Lite (OpenRouter, $0 cost on free tier) produces high-quality Turkish AppleScript pairs at ~92% compile rate

### Phase 2: SFT LoRA Training (RunPod A40, $0.82)

**Qwen3.5 DeltaNet-specific configuration — critical gotchas:**

| Setting | Value | Why |
|---------|-------|-----|
| `target_modules` | `"all-linear"` | DeltaNet has `in_proj_qkv`, `in_proj_z`, `out_proj` — not standard `q_proj/v_proj`. PEFT auto-detects all nn.Linear |
| Quantization | **bf16 only** | Qwen team warns: 4-bit QLoRA causes "higher than normal quantization differences" |
| `attn_implementation` | `"sdpa"` | Flash Attention 2 causes CUDA errors with Qwen3.5 |
| `packing` | `False` | GatedDeltaNet doesn't support packing |
| `group_by_length` | `True` | Alternative to packing for efficient batching |
| `max_length` | 512 | Data max ~414 tokens (NOT `max_seq_length` — TRL 0.29 API change) |

**Results:**
```
Train loss:     0.439 (healthy — 0.5-1.5 range, not overfitting)
Eval loss:      0.539
Token accuracy: 88.7%
Duration:       2h 02m on A40 48GB
Cost:           $0.82
```

**RunPod operational lessons:**
- Always use `--startSSH --ports "22/tcp" --secureCloud` when creating pods
- Spot/community pods fail to initialize — money wasted
- `runpodctl exec python` is unreliable; use SSH directly: `ssh -T -p PORT root@IP`
- Set `TORCH_CUDA_ALLOC_CONF=expandable_segments:True` before training

### Phase 3: GRPO — Skipped

Gate condition: skip if SFT achieves ≥85% compile rate. Got 86%. GRPO not needed.

Additional blocker: `osacompile` (reward signal) is macOS-only, can't run on Linux cloud GPUs.

### Phase 4: Evaluation (86% compile rate)

101 test prompts evaluated (CPU inference, ~43s/prompt):
- **87/101 = 86% compile rate** (target was ≥85%)
- Failures mostly on complex multi-step commands (ZIP, Docker, split pane)
- Simple commands (open app, get property): ~95%+

### Phase 5: ANE Conversion

Standard ANEMLL pipeline with existing Qwen3.5 converter:
```bash
./anemll/utils/convert_model.sh \
    --model models/qwen35-sft-merged \
    --output /tmp/spotlight-ane-output \
    --context 2048 --batch 64 --chunk 4
```

Output: 41 `.mlmodelc` files, 3.8GB, `per_chunk_state: true`, `infer_only: true`

**Known remaining issues:**
1. Swift CLI needs system prompt injection for AppleScript-style output
2. Shape mismatch: batch_size=64 conversion vs single-token inference
3. Tokenizer compatibility: Python 3.9 `transformers` can't load `TokenizersBackend` class — re-download tokenizer with old transformers version

## Prevention / Best Practices

### For future fine-tuning of hybrid DeltaNet models:
1. Always check model card for quantization warnings before choosing QLoRA
2. Use `target_modules="all-linear"` — don't try to enumerate DeltaNet projection names manually
3. Verify `max_length` vs `max_seq_length` against the TRL version installed
4. First Triton kernel compilation step takes 2-5 min — this is normal, not a hang

### For data collection:
1. Start with sdef parsing — gives complete API surface per app
2. `osacompile` is necessary but insufficient — add sdef cross-reference for property validation
3. Synthetic data quality scales with prompt quality, not quantity — use sdef data in the generation prompt
4. 30-line filter removes tutorial fragments and complex handlers that confuse small models

### For RunPod:
1. Always `--startSSH --secureCloud` — no exceptions
2. Pin dependency versions in requirements.txt for reproducibility
3. Download model artifacts BEFORE stopping the pod

## Related Documentation

- `docs/solutions/runtime-errors/ane-mlstate-error-14-coreml-stateful-models.md` — MLState rules for Qwen3.5 ANE
- `docs/solutions/runtime-errors/qwen35-warmup-state-contamination-echo-bug.md` — DeltaNet state management
- `docs/solutions/performance-issues/qwen35-08b-ane-mlstate-optimization.md` — ANE performance (7→20 t/s)
- `docs/brainstorms/2026-03-23-spotlight-ai-macos-agent-brainstorm.md` — Original Spotlight AI concept
- `docs/plans/2026-03-23-feat-spotlight-ai-finetune-pipeline-plan.md` — Detailed implementation plan

## Repository

All code at `~/Desktop2/spotlight-ai/` (14 commits):
- `scripts/parse_sdef.py` — sdef XML parser (18 apps)
- `scripts/import_automator_recipes.py` — macos-automator-mcp importer
- `scripts/import_github_scripts.py` — GitHub .applescript importer
- `scripts/import_scpt_files.py` — macOS .scpt decompiler
- `scripts/expand_templates.py` — EN+TR template expansion
- `scripts/generate_bulk.py` — 50-app synthetic generation (Gemini)
- `scripts/generate_complex.py` — Complex workflow synthetic generation
- `scripts/verify_and_assemble.py` — osacompile verify + JSONL assembly
- `scripts/train_sft.py` — PEFT LoRA training (4 configs A/B/C/D)
- `scripts/merge_adapter.py` — LoRA merge into base model
- `scripts/evaluate_local.py` — osacompile evaluation harness

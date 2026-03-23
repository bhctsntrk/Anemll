---
title: "feat: Spotlight AI Fine-Tuning Pipeline — Data Collection, SFT, GRPO Training"
type: feat
date: 2026-03-23
---

# Spotlight AI Fine-Tuning Pipeline

## Enhancement Summary

**Deepened on:** 2026-03-23
**Focus:** Success criteria, benchmarks, exit criteria for every phase
**Research agents used:** 7 (repo-research, learnings, best-practices x2, framework-docs x2, specflow)

### Key Improvements
1. **Phase-by-phase Go/No-Go flowchart** with concrete numeric thresholds
2. **4-tier evaluation harness** (osacompile → sdef cross-ref → osadecompile roundtrip → execution) — osacompile alone misses property validation
3. **320 benchmark test cases** (40 per app, stratified simple/medium/complex) — frozen before training
4. **Training convergence indicators** with exact thresholds (train loss < 0.2 = memorizing, val gap > 0.3 = overfitting)
5. **GRPO health signals** — KL divergence < 10 nats, reward trajectory monitoring
6. **Forgetting measurement** — perplexity change < 5% on held-out general text
7. **pass@k methodology** — T=0.2 for pass@1, T=0.6 for pass@5, n=20, unbiased estimator formula

### New Considerations Discovered
- osacompile does NOT validate property names — `get bogusproperty of front window` compiles cleanly
- osadecompile roundtrip reveals unresolved references (Tier 3 check)
- Dataset should have 40-70% base model accuracy — too easy = no learning signal
- MLX-LM has no native early stopping — requires custom TrainingCallback
- LoRA inherently mitigates forgetting; >15% perplexity increase = overtraining

---

## Overview

End-to-end pipeline to fine-tune Qwen3.5-0.8B for macOS AppleScript generation. The model will power a Spotlight-like overlay that converts natural language commands to AppleScript and executes them locally on ANE.

**Three core questions this plan answers:**
1. **Veri nereden?** — sdef parsing + macos-automator-mcp recipes + synthetic generation
2. **Reward sim nasil?** — `osacompile` binary reward + format/semantic rewards via GRPO
3. **M1 mi cloud mu?** — SFT local (M1 8GB, free), GRPO local attempt (MLX-GRPO) → cloud fallback (RunPod A100, ~$3-6)

## Problem Statement

Qwen3.5-0.8B runs on ANE at ~20 t/s (MLState pipeline) but has no AppleScript knowledge. We need to:
- Build a high-quality dataset mapping natural language → AppleScript
- Fine-tune the model to reliably generate compilable AppleScript for 8 target apps
- Optionally reinforce with GRPO using verifiable rewards (compilation check)
- Convert back to ANE format without quality loss

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPOTLIGHT AI PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: DATA COLLECTION                                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐              │
│  │ sdef     │  │ automator-mcp│  │ Claude/GPT    │              │
│  │ parsing  │  │ 518 recipes  │  │ synthetic gen │              │
│  └────┬─────┘  └──────┬───────┘  └──────┬────────┘              │
│       │               │                  │                       │
│       └───────────────┼──────────────────┘                       │
│                       ▼                                          │
│              ┌────────────────┐                                   │
│              │ osacompile     │ ← verify ALL pairs compile       │
│              │ verification   │                                   │
│              └───────┬────────┘                                   │
│                      ▼                                           │
│              ┌────────────────┐                                   │
│              │ train.jsonl    │ chat format, 2000-5000 pairs     │
│              │ valid.jsonl    │                                   │
│              │ test.jsonl     │                                   │
│              └───────┬────────┘                                   │
│                      │                                           │
│  Phase 2: SFT (LOCAL — M1 8GB)                                   │
│                      ▼                                           │
│              ┌────────────────┐                                   │
│              │ mlx_lm.lora    │ QLoRA, rank=8, batch=1           │
│              │ Qwen3.5-0.8B  │ grad-checkpoint, mask-prompt      │
│              └───────┬────────┘                                   │
│                      ▼                                           │
│              ┌────────────────┐                                   │
│              │ mlx_lm.fuse    │ merge LoRA → full model          │
│              └───────┬────────┘                                   │
│                      │                                           │
│  Phase 3: GRPO (LOCAL attempt → CLOUD fallback)                  │
│                      ▼                                           │
│              ┌────────────────┐                                   │
│              │ MLX-GRPO       │ osacompile reward                │
│              │ or RunPod A100 │ format + semantic rewards        │
│              └───────┬────────┘                                   │
│                      │                                           │
│  Phase 4: EVALUATION                                             │
│                      ▼                                           │
│              ┌────────────────┐                                   │
│              │ 3-tier harness │ compile → static → execute       │
│              │ pass@1, pass@5 │ per-app breakdown                │
│              └───────┬────────┘                                   │
│                      │                                           │
│  Phase 5: ANE CONVERSION                                         │
│                      ▼                                           │
│              ┌────────────────┐                                   │
│              │ ANEMLL convert │ convert_model.sh → .mlmodelc     │
│              │ + ANE test     │ verify >= 18 t/s maintained      │
│              └────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

---

#### Phase 1: Data Collection (`spotlight-ai/data/`)

**Goal:** 2000-5000 verified instruction→AppleScript pairs

##### 1.1 sdef Parsing (`scripts/parse_sdef.py`)

- [ ] Write Python script to extract commands/parameters from sdef XML
- [ ] Run `sdef /Applications/<App>.app` for each target app
- [ ] Parse XML with `xml.etree.ElementTree`
- [ ] Extract: suite name, command name, parameters (name, type, optional), classes, properties
- [ ] Output structured JSON per app: `data/sdef/<app_name>.json`

**Target apps (MVP 8):**

| App | sdef Source | Expected Commands |
|-----|-----------|-------------------|
| Finder | `/System/Applications/Finder.app` | ~50 (files, folders, windows) |
| Safari | `/Applications/Safari.app` | ~30 (tabs, windows, bookmarks) |
| Mail | `/System/Applications/Mail.app` | ~40 (messages, mailboxes, accounts) |
| Calendar | `/System/Applications/Calendar.app` | ~20 (events, calendars) |
| Notes | `/System/Applications/Notes.app` | ~15 (notes, folders) |
| Music | `/System/Applications/Music.app` | ~35 (tracks, playlists, playback) |
| System Events | `/System/Library/CoreServices/System Events.app` | ~60 (processes, UI, disks) |
| Terminal | `/System/Applications/Utilities/Terminal.app` | ~15 (windows, tabs, settings) |

**Edge case:** Some apps may have empty or minimal sdef files. Fall back to manual AppleScript documentation for these.

```python
# scripts/parse_sdef.py — core logic
import subprocess, xml.etree.ElementTree as ET, json

TARGET_APPS = {
    "Finder": "/System/Applications/Finder.app",
    "Safari": "/Applications/Safari.app",
    "Mail": "/System/Applications/Mail.app",
    # ... etc
}

def parse_app_sdef(app_name, app_path):
    result = subprocess.run(["sdef", app_path], capture_output=True, text=True)
    if result.returncode != 0:
        return {"app": app_name, "error": result.stderr, "commands": []}

    root = ET.fromstring(result.stdout)
    commands = []
    for suite in root.findall('.//suite'):
        for cmd in suite.findall('command'):
            params = [{"name": p.get("name"), "type": p.get("type", ""),
                       "optional": p.get("optional", "no")}
                      for p in cmd.findall("parameter")]
            commands.append({
                "suite": suite.get("name"),
                "name": cmd.get("name"),
                "description": cmd.get("description", ""),
                "parameters": params
            })
    return {"app": app_name, "commands": commands}
```

##### 1.2 macos-automator-mcp Import (`scripts/import_automator_recipes.py`)

- [ ] Clone `steipete/macos-automator-mcp` knowledge_base (518 recipes)
- [ ] Parse Markdown + YAML frontmatter → extract task description + AppleScript code
- [ ] Filter for our 8 target apps
- [ ] Convert to instruction→code pairs
- [ ] Verify each with `osacompile`

**Coverage from automator-mcp:**

| Category | Recipes | Relevant to MVP |
|----------|---------|----------------|
| `07_browsers` (Safari/Chrome) | 73 | ~30 |
| `09_productivity` (Mail/Calendar/Notes) | 64 | ~40 |
| `10_creative` (Music/Spotify) | 66 | ~16 |
| `05_files` (Finder) | 23 | ~20 |
| `06_terminal` | 32 | ~18 |
| `04_system` (System Events) | 27 | ~20 |
| **Total** | **285** | **~144** |

##### 1.3 Template Expansion (`scripts/expand_templates.py`)

- [ ] For each sdef command, create 5-10 natural language variations
- [ ] Include both English and Turkish instructions
- [ ] Vary complexity: simple ("open Safari") → medium ("open Safari and go to apple.com") → complex ("open Safari, create new tab, go to apple.com, bookmark it")
- [ ] Use template strings with slot filling

```python
# Template example
TEMPLATES = {
    "open_app": [
        "{app}'i ac",                    # Turkish simple
        "{app} uygulamasini baslat",     # Turkish formal
        "Open {app}",                    # English simple
        "Launch {app} application",      # English formal
        "Start {app} for me",           # English casual
    ],
    "new_tab": [
        "{app}'de yeni sekme ac",
        "Open a new tab in {app}",
        "Create a new tab in {app}",
    ],
}
```

**Expected yield:** ~1000-1500 template-expanded pairs

##### 1.4 Synthetic Generation (`scripts/generate_synthetic.py`)

- [ ] Use Claude API (or GPT-4) to generate realistic usage scenarios
- [ ] Input: sdef command list + app context
- [ ] Output: diverse instruction→AppleScript pairs
- [ ] Prompt template includes examples of good pairs
- [ ] Generate in batches of 50 per app

```python
# Synthetic generation prompt (simplified)
PROMPT = """Given these AppleScript commands for {app}:
{commands_from_sdef}

Generate {n} diverse natural language instructions with corresponding AppleScript code.
Rules:
- Each instruction should be something a real user would say
- Include both simple and complex commands
- Mix Turkish and English instructions
- Code must be valid, compilable AppleScript
- Format: JSON array of {{"instruction": "...", "code": "..."}}
"""
```

**Expected yield:** ~1000-2000 synthetic pairs
**Cost estimate:** ~$2-5 in API credits (Claude Haiku for volume, Sonnet for quality)

##### 1.5 Verification & Assembly (`scripts/verify_and_assemble.py`)

- [ ] Run `osacompile -o /dev/null` on every code example
- [ ] Remove pairs that fail compilation
- [ ] Deduplicate near-identical instructions (cosine similarity > 0.9)
- [ ] Add difficulty labels: simple/medium/complex
- [ ] Balance dataset across apps (no app > 30% of total)
- [ ] Split 80/10/10 → train/valid/test
- [ ] Format as MLX-LM chat JSONL

```jsonl
{"messages": [{"role": "system", "content": "Generate AppleScript code for the given macOS command. Output only valid AppleScript, no explanations."}, {"role": "user", "content": "Safari'de yeni sekme ac"}, {"role": "assistant", "content": "tell application \"Safari\"\n    activate\n    tell window 1\n        set current tab to (make new tab)\n    end tell\nend tell"}]}
```

- [ ] Validate JSONL format (each line parseable, all fields present)
- [ ] Report statistics: total pairs, per-app counts, compile rate, language distribution

**Final target:** 2000-5000 verified pairs in `data/processed/train.jsonl`

---

#### Phase 2: SFT Fine-Tuning (Local, M1 8GB)

**Goal:** Baseline model with >75% compile rate on test set

##### 2.1 Environment Setup

- [ ] Create project: `mkdir -p ~/Desktop2/spotlight-ai && cd ~/Desktop2/spotlight-ai`
- [ ] Set up training venv (Python 3.11+): `uv venv .venv --python 3.11 && source .venv/bin/activate`
- [ ] Install: `uv pip install "mlx-lm[train]" datasets`
- [ ] Download and quantize base model: `mlx_lm.convert --hf-path Qwen/Qwen3.5-0.8B -q --q-bits 4 --mlx-path models/qwen35-0.8b-4bit`
- [ ] Verify tokenizer chat template: `python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B'); print(t.chat_template[:100])"`

**Note:** ANEMLL conversion (Phase 5) uses the separate Python 3.9 env at `~/Desktop2/anemll-fork/env-anemll/`

##### 2.2 Phase 2a: Small-Scale SFT (500 pairs)

- [ ] Take first 500 highest-quality pairs from assembled dataset
- [ ] Train with conservative settings:

```bash
mlx_lm.lora \
    --model models/qwen35-0.8b-4bit \
    --train \
    --data data/processed/ \
    --batch-size 1 \
    --num-layers 4 \
    --iters 600 \
    --learning-rate 5e-5 \
    --steps-per-report 10 \
    --steps-per-eval 50 \
    --steps-per-save 100 \
    --adapter-path adapters/sft-v1/ \
    --max-seq-length 512 \
    --mask-prompt \
    --grad-checkpoint
```

- [ ] Monitor: training loss should decrease steadily, validation loss should plateau (not increase)
- [ ] Benchmark after 600 iters against test set (100 pairs)
- [ ] Target: >75% compile rate on test set

**Memory estimate:** QLoRA 0.8B + batch 1 + grad-checkpoint ≈ 3-4 GB. M1 8GB'de rahat sigar.

**Time estimate:** ~600 iters with batch 1 ≈ 30-60 min on M1

##### 2.3 Phase 2b: Full-Scale SFT (2000+ pairs)

- [ ] Use full dataset
- [ ] Hyperparameter sweep (run 3-4 configs, pick best validation loss):

**Qwen3.5 DeltaNet Hybrid Mimarisine Özel Notlar:**
- **QLoRA (4-bit) KULLANMA** — Qwen ekibi uyarıyor: quantization farkları normalden yüksek. bf16 veya 8-bit kullan.
- **target_modules="all-linear"** — DeltaNet projections (in_proj_qkv, in_proj_z, out_proj) + attention (q/k/v/o_proj) + MLP hepsi otomatik bulunur
- **Recurrent parametrelere dokunma** — A_log, dt_bias LoRA hedefi olmamalı (zaten nn.Linear değil)
- **Flash Attention 2 kapat** — `--disable_flash_attn2 True` (CUDA error verir)
- **Packing/padding_free desteklenmiyor** — `group_by_length=True` kullan
- **İlk step yavaş** — Triton kernel compilation 2-5 dk sürer, normal

| Config | Rank | Layers | LR | Iters | Quant |
|--------|------|--------|----|-------|-------|
| A (conservative) | 8 | all-linear | 5e-5 | 1000 | bf16 |
| B (default) | 16 | all-linear | 1e-4 | 1000 | bf16 |
| C (aggressive) | 32 | all-linear | 2e-4 | 600 | bf16 |
| D (long) | 16 | all-linear | 5e-5 | 2000 | 8-bit |

- [ ] Pick best config by validation loss (not training loss!)
- [ ] Target: >85% compile rate on test set
- [ ] Merge best adapter: `mlx_lm.fuse --model models/qwen35-0.8b-4bit --adapter-path adapters/best/`

##### 2.4 SFT Quality Check

- [ ] Test fused model with `mlx_lm.generate` on 20 manual prompts
- [ ] Verify model hasn't "forgotten" basic language ability
- [ ] Check for overfitting signs: if test loss >> train loss, reduce iters or increase dropout

---

#### Phase 3: GRPO Reinforcement Learning (Conditional)

**Gate:** Only proceed if SFT plateau at ~80% accuracy. If SFT achieves >90%, skip GRPO.

##### 3.1 Reward Functions

Three reward signals, weighted:

```python
# reward_functions.py

import subprocess

def compile_reward(completions, **kwargs):
    """Binary: does the AppleScript compile? Weight: 1.0"""
    rewards = []
    for comp in completions:
        code = extract_code(comp[0]["content"])
        result = subprocess.run(
            ["osacompile", "-o", "/dev/null"],
            input=code, capture_output=True, text=True, timeout=5
        )
        rewards.append(1.0 if result.returncode == 0 else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Does output start with 'tell application' or valid AS structure? Weight: 0.3"""
    rewards = []
    for comp in completions:
        code = comp[0]["content"].strip()
        valid = code.startswith("tell application") or code.startswith("do shell script")
        rewards.append(0.5 if valid else 0.0)
    return rewards

def app_correctness_reward(completions, prompts, **kwargs):
    """Does the script reference the correct app from the prompt? Weight: 0.5"""
    rewards = []
    for comp, prompt in zip(completions, prompts):
        target_app = extract_target_app(prompt)  # "Safari", "Finder" etc.
        code = comp[0]["content"]
        rewards.append(0.5 if target_app.lower() in code.lower() else 0.0)
    return rewards
```

##### 3.2 Option A: MLX-GRPO (Local, Free)

- [ ] Install: `uv pip install mlx-grpo` (veya clone from Doriandarko/MLX-GRPO)
- [ ] Create config:

```toml
# config/grpo_smoke.toml
[model]
name = "models/qwen35-0.8b-sft-fused"

[training]
group_size = 2          # Low for 8GB
max_completion_length = 128
num_iterations = 100
learning_rate = 1e-6
batch_size = 1

[rewards]
functions = ["compile_reward", "format_reward"]
weights = [1.0, 0.3]
```

- [ ] Run smoke test (100 iters, group_size=2)
- [ ] If OOM → reduce group_size to 1 or max_completion_length to 64
- [ ] If still OOM → proceed to Option B

##### 3.3 Option B: RunPod RTX A6000 (Cloud, ~$1.50)

**GPU:** RTX A6000 — 48GB VRAM, $0.33/hr on-demand. 0.8B QLoRA + group_size=8 rahat sığar.

> **UYARI: Spot ve Community pod'lar initialize olmuyor, boşa para gidiyor.**
> Sadece **Secure Cloud, On-Demand** kullan. Spot/community seçme.

- [ ] Pod oluştur (`runpodctl` ile):

```bash
# RTX A6000, 48GB VRAM, PyTorch template, on-demand secure
runpodctl create pod \
    --name "spotlight-grpo" \
    --gpuType "NVIDIA RTX A6000" \
    --gpuCount 1 \
    --volumeSize 50 \
    --containerDiskSize 20 \
    --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env "JUPYTER_PASSWORD=spotlight2026"

# Pod ID'yi not al
runpodctl get pod
```

- [ ] SSH ile bağlan ve ortam kur:

```bash
# SSH bağlantısı
runpodctl ssh <pod-id>

# Ortam kurulumu
pip install trl peft transformers datasets accelerate

# Model ve veriyi upload et (runpodctl ile)
# Önce local'de:
runpodctl send models/qwen35-0.8b-sft-fused/ <pod-id>:/workspace/model/
runpodctl send data/processed/ <pod-id>:/workspace/data/
runpodctl send scripts/reward_functions.py <pod-id>:/workspace/
```

- [ ] GRPO training:

```bash
python train_grpo.py \
    --model /workspace/model \
    --reward-funcs compile_reward format_reward app_correctness_reward \
    --reward-weights 1.0 0.3 0.5 \
    --num-generations 8 \
    --max-completion-length 256 \
    --per-device-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --num-train-epochs 1 \
    --output-dir /workspace/grpo-output
```

- [ ] Sonuçları indir ve pod'u kapat:

```bash
# Local'de:
runpodctl receive <pod-id>:/workspace/grpo-output/ models/qwen35-grpo/
runpodctl stop pod <pod-id>
runpodctl remove pod <pod-id>
```

**Maliyet tahmini:** 4 saat training = **$1.32**. Budget'ın çok altında.

**Alternatif GPU'lar (on-demand secure):**

| GPU | VRAM | $/hr | 4 saat | Ne zaman |
|-----|------|------|--------|----------|
| **RTX A6000** | 48 GB | $0.33 | $1.32 | **Tavsiye — best value** |
| A40 | 48 GB | $0.35 | $1.40 | A6000 yoksa |
| RTX 5000 Ada | 32 GB | $0.49 | $1.96 | Daha yeni arch istersen |
| RTX 4000 Ada | 20 GB | $0.20 | $0.80 | En ucuz, group_size=2-4 |

**Important:** `osacompile` is macOS-only. On RunPod (Linux), reward simulation options:
1. **Regex-based syntax check** — approximate `osacompile` with pattern matching (lossy but functional for GRPO)
2. **Pre-compute rewards**: Generate completions on RunPod → download → score with `osacompile` on Mac → upload scored data → continue training
3. **osascript Docker**: There is NO reliable AppleScript runtime on Linux

**Recommended:** Option 1 for simplicity, Option 2 for accuracy. Or stick with MLX-GRPO locally where `osacompile` is native.

##### 3.4 GRPO vs. Rejection Sampling Alternative

If GRPO is too complex, a simpler RL-free alternative:

- [ ] Generate N completions per prompt from SFT model
- [ ] Score with `osacompile` (compile = keep, fail = discard)
- [ ] Re-train SFT on the filtered "best-of-N" completions
- [ ] Iterate 2-3 times

This is **rejection sampling fine-tuning** (RFT) — simpler than GRPO, runs entirely on M1, and often achieves 80-90% of GRPO's improvement.

---

#### Phase 4: Evaluation (`scripts/evaluate.py`)

**Goal:** Rigorous measurement of model quality

##### 4.1 Test Harness — 4-Tier Evaluation

- [ ] Build 4-tier evaluation pipeline:

```python
# scripts/evaluate.py

import subprocess, xml.etree.ElementTree as ET

def tier1_compile(code):
    """Tier 1: osacompile — syntax + class/verb validation (~100ms)
    Catches: syntax errors, unknown classes, unknown command verbs
    MISSES: invalid property names, wrong parameter types, non-existent apps"""
    result = subprocess.run(["osacompile", "-o", "/dev/null"],
                           input=code, capture_output=True, text=True, timeout=5)
    return result.returncode == 0, result.stderr

def tier2_sdef_validate(code, expected_app, sdef_db):
    """Tier 2: sdef cross-reference — property/element validation (~10ms)
    Catches: invalid property names that osacompile misses
    Example: 'get bogusproperty of front window' compiles but fails here"""
    # Check app name in tell block
    has_app = f'application "{expected_app}"' in code
    # Cross-reference properties against app's sdef
    invalid_props = sdef_db.validate_properties(code, expected_app)
    return has_app and len(invalid_props) == 0, invalid_props

def tier3_roundtrip(code):
    """Tier 3: compile-decompile roundtrip — unresolved reference detection (~150ms)
    If osacompile can't resolve a term against the app dictionary,
    the decompiled form will differ from input (raw identifier vs canonical)"""
    compile_result = subprocess.run(["osacompile", "-o", "/tmp/check.scpt"],
                                   input=code, capture_output=True, text=True, timeout=5)
    if compile_result.returncode != 0:
        return False, "compile failed"
    decompile_result = subprocess.run(["osadecompile", "/tmp/check.scpt"],
                                     capture_output=True, text=True, timeout=5)
    # Normalize whitespace for comparison
    return True, decompile_result.stdout.strip()

def tier4_execute(code, timeout=10):
    """Tier 4: osascript execution — READ-ONLY commands only (~1-10s)
    SAFETY: only execute pre-approved whitelist of safe read-only prompts
    Never execute: set, make, delete, move, send, do script, keystroke"""
    result = subprocess.run(["osascript", "-e", code],
                           capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0, result.stdout, result.stderr
```

##### 4.2 Metrics

**Primary (reported per-app and overall):**
- [ ] **compile_rate (Tier 1)**: % that pass osacompile
- [ ] **sdef_valid_rate (Tier 2)**: % with valid property/element references
- [ ] **pass@1 (T=0.2, n=20)**: Primary deployment metric — does first generation work?
- [ ] **pass@5 (T=0.6, n=20)**: Model capability ceiling

**Secondary:**
- [ ] **app_accuracy**: % that reference the correct target app
- [ ] **roundtrip_match (Tier 3)**: % where decompiled form matches input structure
- [ ] **exec_pass (Tier 4)**: % of read-only test cases that execute without error

**Results table format** (generated by `scripts/evaluate.py --report`):

| App | Tier1 compile | Tier2 sdef | pass@1 | pass@5 | App correct |
|-----|--------------|------------|--------|--------|-------------|
| Finder | ?% | ?% | ?% | ?% | ?% |
| Safari | ?% | ?% | ?% | ?% | ?% |
| Mail | ?% | ?% | ?% | ?% | ?% |
| Calendar | ?% | ?% | ?% | ?% | ?% |
| Notes | ?% | ?% | ?% | ?% | ?% |
| Music | ?% | ?% | ?% | ?% | ?% |
| System Events | ?% | ?% | ?% | ?% | ?% |
| Terminal | ?% | ?% | ?% | ?% | ?% |
| **Overall** | **?%** | **?%** | **?%** | **?%** | **?%** |

Plus per-difficulty breakdown: Simple / Medium / Complex

##### 4.3 Benchmark Suite (`data/benchmark/`) — 320 Test Cases

- [ ] 320 test cases, 40 per app, stratified:
  - 128 simple (16 per app): single tell block, one command (e.g. "Get the URL of the current Safari tab")
  - 128 medium (16 per app): 2-5 commands, loops, filters (e.g. "Find unread emails from X and mark as read")
  - 64 complex (8 per app): multi-app, error handling, dynamic (e.g. "Get Safari URL, create Calendar event with it")
- [ ] Include both English and Turkish variants (70/30 ratio)
- [ ] Each test case includes: instruction, expected app, difficulty label, expected AppleScript (reference solution)
- [ ] **FREEZE benchmark before Phase 2 starts** — no modifications after training begins
- [ ] Run both base model AND fine-tuned model on same benchmark for comparison
- [ ] Store results in `results/eval_*.json` with full per-prompt breakdown

**Sample benchmark prompts (5 per app, 40 seed prompts):**

<details>
<summary>Finder examples</summary>

| # | Difficulty | Prompt | Expected key pattern |
|---|-----------|--------|---------------------|
| 1 | Simple | "Get the name of every file on the Desktop" | `get name of every file of desktop` |
| 2 | Simple | "Check free space on startup disk" | `get free space of startup disk` |
| 3 | Medium | "Find all .txt files on Desktop, move to Documents" | `whose name extension is "txt"` + `move` |
| 4 | Medium | "Create folder 'Archive' on Desktop if not exists" | `if not (exists folder` + `make new folder` |
| 5 | Complex | "Count files in each Desktop folder, return as records" | `repeat with` + `count of files` |

</details>

<details>
<summary>Safari examples</summary>

| # | Difficulty | Prompt | Expected key pattern |
|---|-----------|--------|---------------------|
| 1 | Simple | "Get URL of current tab" | `URL of current tab of front window` |
| 2 | Simple | "Get page title of every open tab" | `name of every tab` |
| 3 | Medium | "Close all tabs except current one" | `repeat` + `close tab` + index check |
| 4 | Medium | "Open 3 URLs in new tabs" | `make new tab` + `set URL` |
| 5 | Complex | "Get URL+title of every tab across all windows" | Nested `repeat with w` + `repeat with t` |

</details>

<details>
<summary>Mail, Calendar, Notes, Music, System Events, Terminal</summary>

See `data/benchmark/README.md` for full 320-prompt benchmark with reference solutions.
Each app follows the same pattern: 16 simple, 16 medium, 8 complex.

</details>

##### 4.4 Regression Testing

- [ ] Test 20 general knowledge questions (non-AppleScript) to verify no catastrophic forgetting
- [ ] Compare perplexity on general text before/after fine-tuning

---

#### Phase 5: ANE Conversion & Deployment

**Goal:** Fine-tuned model running on ANE at >= 18 t/s (MLState pipeline)

##### 5.1 Model Preparation

- [ ] Export fused model to HuggingFace format (if MLX format):
  ```bash
  # mlx_lm.fuse already outputs HF-compatible format
  # Verify: config.json, model.safetensors, tokenizer.json exist in fused_model/
  ```

##### 5.2 ANEMLL Conversion

- [ ] Run existing conversion pipeline:
  ```bash
  cd ~/Desktop2/anemll-fork
  ./anemll/utils/convert_model.sh \
      --model ~/Desktop2/spotlight-ai/models/qwen35-0.8b-sft-fused \
      --output ~/Desktop2/spotlight-ai/models/ane-output \
      --context 2048 \
      --batch 64 \
      --chunk 4
  ```
- [ ] If conversion fails: check config.json `model_type` matches `qwen3_5`
- [ ] Known issue: `handle_unused_inputs` crash — monkey-patch already in qwen3_5_converter.py

##### 5.3 Verification

- [ ] Test with simple_chat.py:
  ```bash
  python tests/simple_chat.py --meta ~/Desktop2/spotlight-ai/models/ane-output/meta.yaml
  ```
- [ ] Verify: same questions from benchmark produce compilable AppleScript
- [ ] Measure inference speed: should maintain ~20 t/s (MLState pipeline, fine-tuning doesn't change architecture)
- [ ] Compare outputs: MLX inference vs ANE inference on same prompts (should be near-identical)

---

## Decision: M1 mi Cloud mu?

| Stage | Where | Cost | Why |
|-------|-------|------|-----|
| Data collection | M1 (local) | ~$2-5 API | sdef parsing + osacompile are macOS-only |
| SFT (LoRA) | M1 (local) | Free | QLoRA 0.8B fits in 3-4 GB, MLX-LM native |
| GRPO | M1 first | Free | MLX-GRPO + osacompile native. Try first |
| GRPO fallback | RunPod RTX A6000 | ~$1.30 | Only if M1 OOMs. On-demand secure only (spot/community initialize olmaz!) |
| Evaluation | M1 (local) | Free | osacompile + osascript macOS-only |
| ANE conversion | M1 (local) | Free | ANEMLL pipeline + coremltools |

**Total budget: ~$3-7** (API credits ~$2-5 + RunPod RTX A6000 ~$1.30)

## Alternative Approaches Considered

### A. Full Cloud Training (Rejected)
RunPod/Lambda for everything. Problem: `osacompile` is macOS-only. Can't verify AppleScript on Linux. Would need pre-computed rewards.

### B. Larger Model — Qwen3.5-2B (Rejected)
More capacity but doesn't fit on M1 8GB for ANE inference (error -14, 208MB state). Defeats the purpose of local deployment.

### C. Template-Based (No ML) Fallback
If fine-tuning fails entirely: regex matching + template AppleScript. Less flexible but 100% reliable for supported commands. Keep as safety net.

### D. Rejection Sampling Instead of GRPO
Simpler alternative to GRPO — generate N samples, keep compilable ones, retrain. Could replace Phase 3 entirely if GRPO complexity isn't worth it.

## Critical Notes (from SpecFlow Analysis)

### 1. Python Version Conflict
- **Training (MLX-LM, MLX-GRPO):** Python 3.11+ required
- **Conversion (ANEMLL, coremltools):** Python 3.9 required
- **Solution:** Two separate venvs. Training in `spotlight-ai/.venv` (3.11+), conversion in `anemll-fork/env-anemll` (3.9). Fused model saved to disk as handoff point.

### 2. osacompile is macOS-Only
Cloud GRPO (RunPod) cannot run `osacompile`. Three options:
- **(a) Local GRPO only** — MLX-GRPO on M1 with aggressive memory settings (recommended first attempt)
- **(b) Rejection sampling** — generate on cloud, download, score locally, retrain. No RL needed.
- **(c) Regex approximation** — approximate `osacompile` with syntax checks on Linux (lossy but functional)

### 3. Model Download Step
`/tmp/qwen35-0.8b/` is volatile and gone. Add explicit download step:
```bash
# In training venv (3.11+)
mlx_lm.convert --hf-path Qwen/Qwen3.5-0.8B -q --q-bits 4 --mlx-path models/qwen35-0.8b-4bit
```

### 4. Chat Template Consistency
Training must use Qwen3.5's native ChatML template from `tokenizer_config.json`. If training template differs from inference template in ANEMLL, outputs will degrade. Always use `--mask-prompt` to train only on assistant tokens.

### 5. Execution Sandbox for Evaluation
Tier-3 evaluation (osascript execution) can delete files, send emails, etc. Rules:
- Only execute on a pre-approved **whitelist** of safe read-only commands
- Destructive commands (rm, delete, send, create) → compile-only verification
- Use `timeout 10` for all osascript calls

### 6. sdef Parsing Depth
sdef files are not flat — they have nested suites, classes, properties, elements, enumerations. The parser must extract:
- Commands (with parameters) — `open`, `close`, `make`, `delete`
- Classes and their properties — `window.name`, `tab.URL`
- Enumerations — valid values for parameters
- Inheritance — `item` → `file` → `alias file`

Property access patterns like `name of every window` are as important as commands.

### 7. Inference Speed Target
The correct target is **~20 t/s** (MLState pipeline), not 8 t/s. The 8 t/s figure was from the old Regular I/O pipeline. Fine-tuning doesn't change architecture, so speed should be maintained. A regression to <15 t/s indicates a conversion problem.

## Dependencies & Prerequisites

| Dependency | Status | Notes |
|-----------|--------|-------|
| Qwen3.5-0.8B on ANE | Done | ANEMLL PR #50, ~8 t/s |
| MLX-LM | Available | v0.31.1, `pip install "mlx-lm[train]"` |
| macos-automator-mcp | Available | 518 recipes, MIT license |
| Claude/GPT API | Need key | For synthetic data generation |
| RunPod account | Optional | Only if GRPO doesn't fit on M1 |
| MLX-GRPO | Available | github.com/Doriandarko/MLX-GRPO |
| coremltools | Available | >=8.2, monkey-patch for state inputs |

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 0.8B too small for AppleScript quality | Low | High | Focus on simple+medium commands (80% of use). Template fallback for complex |
| M1 OOMs during GRPO | Medium | Low | MLX-GRPO smoke test first. Fallback: rejection sampling (no RL needed) |
| Fine-tuned model breaks ANE conversion | Very Low | High | ANEMLL already converts Qwen3.5 fine-tuned models (tested with DeepHermes). LoRA merge preserves architecture |
| Turkish instructions degrade English quality | Low | Medium | Keep 70% English / 30% Turkish ratio. Test both languages separately |
| osacompile passes but script fails at runtime | Medium | Low | Tier 3 evaluation catches this. Execution testing on safe commands |
| Dataset too small / overfit | Low | Medium | Start with 500 pairs (Phase 2a). Scale to 5000 only if needed |
| sdef files empty for some apps | Low | Low | Supplement with manual documentation + automator-mcp recipes |
| Catastrophic forgetting (general ability lost) | Low | Medium | Regression test with 20 general knowledge questions. LoRA preserves most base knowledge |

## Success Metrics & Exit Criteria (Deepened)

### Phase 1 Exit Criteria: Data Quality

| Metric | Threshold | How to Measure |
|--------|-----------|----------------|
| Total verified pairs | >= 2000 | `wc -l data/processed/train.jsonl` |
| osacompile pass rate | **100%** | Every pair in train/valid/test must compile |
| Per-app minimum | >= 40 pairs per app | `jq` count per app field |
| Per-app maximum | <= 30% of total | No single app dominates |
| Dedup (MinHash) | Jaccard threshold 0.7, ngram=5, perm=256 | Removes ~20-50% of raw data |
| Semantic dedup (cosine) | >= 0.95 removed | Using base model embeddings |
| Language ratio | 70% EN / 30% TR | Count per language field |
| Difficulty balance | 50% simple, 30% medium, 20% complex | Per difficulty label |
| Base model accuracy on train set | 40-70% | Run base model on 100 random train pairs. If >80% already correct, data too easy (no learning signal) |

**Concrete check:** After assembly, run this validation:
```bash
# Must print "ALL PASS"
python scripts/verify_and_assemble.py --validate-only --data data/processed/
```

### Phase 2 Exit Criteria: SFT Training

#### Training Convergence Indicators

| Signal | Healthy | Warning | Stop |
|--------|---------|---------|------|
| Train loss | Steady decrease → plateau at 0.5-1.5 | Below 0.2 (memorizing) | Rising after plateau |
| Val loss | Tracks train loss within ~15% | Plateaus while train drops | Increases for 3+ consecutive evals |
| Train/val gap | < 0.3 absolute | 0.3-0.5 | > 0.5 |

**Early stopping config** (via custom TrainingCallback):
- `patience = 4` consecutive evals without improvement
- `min_delta = 0.01`
- `steps_per_eval = 50` (evaluate frequently)

**MLX-LM output to capture** (JSON export via TrainingCallback):
```
Iter N: Train loss X.XXX, Learning Rate X.XXe-XX, It/sec X.XX, Tokens/sec XXX, Peak mem X.XX GB
Iter N: Val loss X.XXX, Val took X.XXs
```
- Perplexity = exp(val_loss). Loss 1.0 = PPL 2.72, Loss 0.5 = PPL 1.65

#### Phase 2a Exit Criteria (500 pairs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Val loss | Plateaued (not rising) by iter 400 | TrainingCallback JSON |
| Compile rate on test set | **>= 75%** | `scripts/evaluate.py --tier compile` |
| sdef-valid rate | >= 65% | `scripts/evaluate.py --tier static` |
| App correctness | >= 80% | Correct app in tell block |
| Training time | <= 60 min | Wall clock on M1 |

#### Phase 2b Exit Criteria (2000+ pairs, best config)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Val loss | Lower than Phase 2a best | Compare across 4 configs |
| Compile rate (pass@1) | **>= 85%** | T=0.2, n=20 samples per prompt |
| sdef-valid rate | >= 75% | Tier 2 static check |
| App correctness | >= 90% | |
| Simple commands pass@1 | >= 95% | On 128 simple benchmark prompts |
| Medium commands pass@1 | >= 80% | On 128 medium benchmark prompts |
| Complex commands pass@1 | >= 50% | On 64 complex benchmark prompts |
| Forgetting (perplexity) | < 5% increase on held-out general text | Compare PPL base vs fused |

**Config selection:** Pick by lowest validation loss, NOT training loss. Break ties with compile rate on test set.

### Phase 3 Gate & Exit Criteria: GRPO

**Entry gate:** Only proceed if Phase 2b compile rate < 90%. If >= 90%, skip to Phase 4.

#### GRPO Training Health Indicators

| Signal | Healthy | Diverging |
|--------|---------|-----------|
| Reward mean | Stable upward trend | Wild oscillation or flat |
| KL divergence | Gradual increase, **< 10 nats** | Unbounded growth (> 15 nats) |
| Response length | Steady or slight increase | Sudden explosion or collapse |
| Compile rate (periodic eval) | Improving | Declining (reward hacking) |

#### Phase 3 Exit Criteria

| Metric | Target | vs Phase 2b |
|--------|--------|-------------|
| Compile rate (pass@1) | **>= 90%** | +5 pp minimum improvement |
| sdef-valid rate | >= 85% | |
| Simple commands pass@1 | >= 98% | |
| Medium commands pass@1 | >= 85% | |
| KL divergence | < 10 nats | |
| Forgetting (perplexity) | < 10% increase | Slightly more forgetting acceptable |

**Expected improvement from GRPO:** +5-20 percentage points over SFT (based on literature: Qwen2.5-0.5B GSM8K went 21%→37.5% with GRPO).

### Phase 4 Exit Criteria: Evaluation

#### Benchmark Suite: 320 Test Cases

| App | Simple | Medium | Complex | Total |
|-----|--------|--------|---------|-------|
| Finder | 16 | 16 | 8 | 40 |
| Safari | 16 | 16 | 8 | 40 |
| Mail | 16 | 16 | 8 | 40 |
| Calendar | 16 | 16 | 8 | 40 |
| Notes | 16 | 16 | 8 | 40 |
| Music | 16 | 16 | 8 | 40 |
| System Events | 16 | 16 | 8 | 40 |
| Terminal | 16 | 16 | 8 | 40 |
| **Total** | **128** | **128** | **64** | **320** |

**Freeze benchmark BEFORE any training begins.** No changes allowed after Phase 2 starts.

#### 4-Tier Evaluation Harness

| Tier | What It Checks | Cost | Catches |
|------|---------------|------|---------|
| 1: osacompile | Syntax + class/verb names | ~100ms | ~30% of errors |
| 2: sdef cross-ref | Property names, element types, inheritance | ~10ms (Python XML) | ~25% additional |
| 3: osadecompile roundtrip | Unresolved references (compile→decompile→compare) | ~150ms | ~10% additional |
| 4: osascript execution | Runtime errors (read-only commands ONLY) | ~1-10s | Runtime failures |

**Important osacompile gap discovered:** osacompile does NOT validate property names or parameter types. `get bogusproperty of front window` compiles cleanly. Tier 2 (sdef cross-ref) catches this.

#### pass@k Evaluation Settings

| Metric | Temperature | Top-p | Samples (n) | Purpose |
|--------|-------------|-------|-------------|---------|
| **pass@1** (primary) | 0.2 | 0.95 | 20 per prompt | Production quality |
| pass@5 | 0.6 | 0.95 | 20 per prompt | Model capability ceiling |

**Formula (unbiased estimator):**
```python
def pass_at_k(n, c, k):
    """n=total samples, c=correct samples, k=budget"""
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

#### Regression Testing

| Metric | Threshold | Corpus |
|--------|-----------|--------|
| Perplexity change | < 5% increase (excellent), < 10% (acceptable) | 1K samples from WikiText-103 |
| 20 general knowledge questions | Qualitative check — answers coherent | Manual review |

**If perplexity increase > 15%:** Retrain with lower rank or fewer iterations. LoRA inherently mitigates forgetting, so >15% indicates overtraining.

### Phase 5 Exit Criteria: ANE Conversion

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Conversion success | No errors | convert_model.sh exit code 0 |
| ANE inference speed | **>= 18 t/s** (allowing minor regression from 20 t/s) | `simple_chat.py` timing |
| MLX vs ANE output match | >= 95% token match on 50 prompts | Token-by-token comparison script |
| Compile rate on benchmark | Within 2 pp of MLX results | Same 320 benchmark, ANE inference |
| Memory usage | < 4 GB peak | Activity Monitor during inference |

**Speed regression > 25% (below 15 t/s):** Indicates conversion problem. Check chunk count, state buffer sizes.

### Summary: Phase-by-Phase Go/No-Go

```
Phase 1 ──► compile rate = 100%, pairs >= 2000, per-app >= 40
              │ FAIL → add more data sources, fix compilation errors
              ▼ PASS
Phase 2a ──► compile rate >= 75%, val loss plateaued
              │ FAIL → check data quality, adjust hyperparams
              ▼ PASS
Phase 2b ──► compile rate >= 85%, forgetting < 5%
              │ FAIL → try different LoRA configs, add data
              ▼ PASS
              │
              ├── compile rate >= 90%? ──YES──► Skip Phase 3, go to Phase 4
              │
              ▼ NO
Phase 3 ───► compile rate >= 90%, KL < 10 nats
              │ FAIL → try rejection sampling (Phase 3.4)
              ▼ PASS
Phase 4 ───► full benchmark run, regression check
              │ FAIL → iterate on data/training
              ▼ PASS
Phase 5 ───► ANE speed >= 18 t/s, output match >= 95%
              │ FAIL → check conversion, chunk count
              ▼ PASS ── SHIP IT
```

## File Structure

```
spotlight-ai/
├── .venv/                          # uv managed
├── scripts/
│   ├── parse_sdef.py               # Phase 1.1: sdef → JSON
│   ├── import_automator_recipes.py # Phase 1.2: mcp → pairs
│   ├── expand_templates.py         # Phase 1.3: template expansion
│   ├── generate_synthetic.py       # Phase 1.4: Claude/GPT synthetic
│   ├── verify_and_assemble.py      # Phase 1.5: osacompile + JSONL
│   ├── evaluate.py                 # Phase 4: 3-tier evaluation
│   └── reward_functions.py         # Phase 3: GRPO rewards
├── data/
│   ├── sdef/                       # Raw sdef JSON per app
│   ├── raw/                        # Unverified pairs from all sources
│   ├── processed/                  # train.jsonl, valid.jsonl, test.jsonl
│   └── benchmark/                  # 150 held-out test cases
├── models/
│   ├── qwen35-0.8b-4bit/          # Base model (quantized)
│   ├── adapters/                   # LoRA checkpoints
│   │   ├── sft-v1/
│   │   └── best/
│   ├── qwen35-0.8b-sft-fused/     # Merged SFT model
│   └── ane-output/                 # ANEMLL converted .mlmodelc
├── config/
│   ├── grpo_smoke.toml             # MLX-GRPO config
│   └── sft_config.yaml             # MLX-LM LoRA config
└── results/
    ├── eval_base.json              # Base model benchmark
    ├── eval_sft_v1.json            # SFT Phase 2a benchmark
    ├── eval_sft_v2.json            # SFT Phase 2b benchmark
    └── eval_grpo.json              # GRPO benchmark (if done)
```

## Execution Order

```
Phase 1 (Data) ──────────────────────── ~1-2 days
  1.1 sdef parsing                      ~2 hours
  1.2 automator-mcp import              ~2 hours  (parallel with 1.1)
  1.3 template expansion                ~3 hours
  1.4 synthetic generation              ~4 hours  (parallel with 1.3, needs API)
  1.5 verify + assemble                 ~1 hour

Phase 2a (SFT small) ───────────────── ~half day
  2.1 environment setup                 ~30 min
  2.2 train 500 pairs, 600 iters        ~1 hour
  2.2 evaluate                          ~30 min

Phase 2b (SFT full) ────────────────── ~1 day
  2.3 hyperparameter sweep (4 configs)  ~4 hours
  2.3 evaluate best                     ~30 min
  2.4 quality check                     ~30 min

Phase 3 (GRPO — conditional) ───────── ~1 day
  3.2 MLX-GRPO smoke test               ~2 hours
  3.3 full GRPO (local or cloud)        ~4 hours

Phase 4 (Evaluation) ───────────────── ~half day
  4.1-4.4 full benchmark + regression   ~2 hours

Phase 5 (ANE conversion) ───────────── ~half day
  5.1-5.3 convert + verify              ~2 hours
```

**Total: ~4-5 days of focused work**

## References & Research

### Internal
- Brainstorm: `docs/brainstorms/2026-03-23-spotlight-ai-macos-agent-brainstorm.md`
- ANE MLState lessons: `docs/solutions/runtime-errors/ane-mlstate-error-14-coreml-stateful-models.md`
- Warmup contamination: `docs/solutions/runtime-errors/qwen35-warmup-state-contamination-echo-bug.md`
- ANEMLL conversion: `anemll/utils/convert_model.sh`, `anemll/ane_converter/qwen3_5_converter.py`
- Simple chat (for testing): `tests/simple_chat.py`

### External
- [MLX-LM LoRA Documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [macos-automator-mcp](https://github.com/steipete/macos-automator-mcp) — 518 AppleScript recipes
- [ASUnit](https://github.com/lifepillar/ASUnit) — AppleScript unit testing
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [RunPod Pricing](https://www.runpod.io/pricing) — A100 40GB ~$0.89/hr
- [Apple sdef DTD](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ScriptableCocoaApplications/SApps_creating_sdef/SAppsCreateSdef.html)

# Agent Requirements Guide

This document summarizes the key requirements and rules collected from previous discussions for implementing Qwen 3 model support in this repository.

## General Principles

- **No modifications to `llama_model.py`**. The existing Llama model implementation must remain untouched as it is already functional.
- **Do not subclass Qwen models from Llama**. Instead, copy relevant logic and rename classes and variables from "Llama" to "Qwen".
- All dense layers must be expressed as `nn.Conv2d` with `kernel_size=(1,1)` to comply with ANE constraints.
- Keep tensor ranks ≤4 with dimensions `(N, C, H, W)`. Do not exceed 16 384 elements in height or width, and 65 536 elements in the channel dimension.
- Maintain a trailing dimension ≥64 wherever possible for better tiling on ANE.
- Follow LM‑head width slicing as in `llama_model.py` (see `slice_lm_head`) so that weight matrices larger than 16 384 in width are split accordingly.

## Functional Goals

1. **Model Definition**
   - Provide `anemll/models/qwen_model.py` with embeddings, rotary positional encodings, multi‑head attention, MLP blocks and RMSNorm.
   - Ensure weight loading from Hugging Face state dicts, performing necessary reshape and transpose operations (see the Transformers implementation at `transformers/models/qwen3/modeling_qwen3.py`).
   - Implement a runtime companion file `cpp/qwen_model.cpp` (Objective‑C++/C++) similar to the existing Llama runtime.

2. **Conversion Pipeline**
   - Extend the conversion utilities (Python helpers and `convert_model.sh`) to recognize and process Qwen 3 checkpoints via a `QwenConverter`.
   - The pipeline must work for the 0.6 B model and scale to the 8 B model.

3. **Testing**
   - Provide unit tests verifying shape parity between HF and ANEMLL tensors and deterministic inference on short prompts.
   - Ensure tests run via `pytest` without requiring internet access and succeed when Torch is available.

4. **Documentation**
   - Add `docs/qwen3.md` describing usage and known limits.
   - Update `README.md` with Qwen 3 support in the conversion matrix.

5. **Other Rules**
   - Remove placeholder `.h` and `.cpp` files if they do not implement functionality.
   - Keep code formatted with `ruff` and `black`; no linter warnings should remain.

These guidelines capture the latest requirements for adding Qwen 3 model support while adhering to Apple Neural Engine constraints.

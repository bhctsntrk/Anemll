import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from anemll.models.qwen_model import (
    QwenConfig,
    QwenForCausalLM,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from test_py_llama import make_causal_mask


@torch.no_grad()
def test_qwen_hf_vs_custom_inference():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=MODEL_DTYPE)
    hf_model.eval()
    hf_model = hf_model.to(TEST_DEVICE)

    prompt = "what is capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(TEST_DEVICE)

    hf_logits = hf_model(**inputs).logits
    hf_next = torch.argmax(hf_logits[:, -1, :], dim=-1)

    local_dir = snapshot_download(model_name)
    config = QwenConfig(**hf_model.config.to_dict())
    custom_model = QwenForCausalLM(config).to(TEST_DEVICE)
    assert custom_model.load_pretrained_weights(local_dir)
    custom_model.eval()

    seq_len = inputs["input_ids"].shape[1]
    causal_mask = make_causal_mask(seq_len, 0)
    position_ids = torch.arange(seq_len, device=TEST_DEVICE).unsqueeze(0)
    update_mask = torch.zeros(
        (1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
    )
    current_pos = torch.tensor([0], device=TEST_DEVICE)

    custom_logits = custom_model(
        input_ids=inputs["input_ids"],
        update_mask=update_mask,
        position_ids=position_ids,
        causal_mask=causal_mask,
        current_pos=current_pos,
        IN_PREFILL=False,
    )

    custom_next = torch.argmax(custom_logits[:, -1, :], dim=-1)

    assert torch.equal(hf_next, custom_next)

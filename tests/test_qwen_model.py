import torch

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


def test_qwen_forward_small():
    config = QwenConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=8,
        vocab_size=100,
    )
    model = QwenForCausalLM(config, use_ane_norm=False, enable_coreml=False).to(
        TEST_DEVICE
    )
    model.eval()

    batch_size = 1
    seq_length = 1
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_length), device=TEST_DEVICE
    )
    update_mask = torch.zeros(
        (batch_size, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
    )
    position_ids = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
    causal_mask = make_causal_mask(seq_length, 0)
    current_pos = torch.tensor([0], device=TEST_DEVICE)
    single_causal_mask = causal_mask[
        :, :, current_pos : current_pos + 1, :CONTEXT_LENGTH
    ]

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=single_causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False,
        )

    assert output.shape == (batch_size, seq_length, config.vocab_size)

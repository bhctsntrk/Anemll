"""Converter for Qwen 3 models.

This module provides a lightweight converter that mirrors the
:class:`LlamaConverter` behaviour for Qwen models without inheriting from
it. Only the pieces required for the unit tests are implemented."""

from __future__ import annotations

import numpy as np
import torch
import coremltools as ct

from .base_converter import BaseConverter
from ..models.qwen_model import (
    QwenForCausalLM,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
)


class QwenConverter(BaseConverter):
    """Handle conversion of Qwen 3 models to Core ML."""

    model_cls = QwenForCausalLM

    def __init__(
        self,
        model: QwenForCausalLM,
        context_length: int = CONTEXT_LENGTH,
        batch_size: int = 64,
        lut_bits: int | None = 4,
        per_channel: int = 8,
    ) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.converted_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(self) -> ct.models.MLModel:
        """Convert the wrapped model to CoreML format."""
        self.preprocess()
        mlmodel = self.convert_to_coreml(self.model)
        self.postprocess()
        return mlmodel

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def convert_to_coreml(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert the entire model to CoreML."""

        class Wrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM) -> None:
                super().__init__()
                self.model = model

            def forward(
                self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
            ) -> torch.Tensor:
                return self.model(
                    input_ids=input_ids,
                    update_mask=torch.zeros(
                        (1, 1, CONTEXT_LENGTH, 1),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    IN_PREFILL=False,
                )

        wrapper = Wrapper(model)
        wrapper.eval()

        sample_input_ids = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
        sample_position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        sample_causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )
        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        traced = torch.jit.trace(
            wrapper,
            (
                sample_input_ids,
                sample_position_ids,
                sample_causal_mask,
                sample_current_pos,
            ),
        )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="input_ids", shape=sample_input_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="position_ids", shape=sample_position_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="current_pos", shape=sample_current_pos.shape, dtype=np.int32
                ),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        return mlmodel

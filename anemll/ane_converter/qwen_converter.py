"""Converter for Qwen 3 models.

This module provides a lightweight converter that mirrors the
:class:`LlamaConverter` behaviour for Qwen models without inheriting from
it. Only the pieces required for the unit tests are implemented."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
import coremltools as ct

from .base_converter import BaseConverter
from ..models.qwen_model import (
    QwenForCausalLM,
    QwenConfig,
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
        self.head_dim = model.model.config.hidden_size // model.model.config.num_attention_heads
        self.converted_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(self) -> ct.models.MLModel:
        """Convert the wrapped model to CoreML format."""
        print("QwenConverter.convert() called")
        print("Calling preprocess()...")
        self.preprocess()
        print("Calling convert_to_coreml()...")
        mlmodel = self.convert_to_coreml(self.model)
        print("Calling postprocess()...")
        self.postprocess()
        print("QwenConverter.convert() completed")
        return mlmodel

    def convert_to_coreml(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert the entire model to CoreML."""
        print("Creating wrapper model...")

        class Wrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM, context_length: int) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length

            def forward(
                self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
                update_mask: torch.Tensor,
            ) -> torch.Tensor:
                # Fixed window approach: return full logits, extract position on Python side
                return self.model(
                    input_ids=input_ids,
                    update_mask=update_mask,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    IN_PREFILL=False,
                )

        wrapper = Wrapper(model, self.context_length)
        wrapper.eval()
        print("Wrapper model created and set to eval mode")

        print("Preparing model inputs for tracing...")
        # Use fixed window approach - full context length with padding
        sample_input_ids = torch.zeros((1, self.context_length), dtype=torch.int32, device=TEST_DEVICE)  # [1, context_length] - full window
        sample_position_ids = torch.arange(self.context_length, dtype=torch.int32, device=TEST_DEVICE)  # [context_length] - [0, 1, 2, ..., 255]
        sample_causal_mask = torch.zeros((1, 1, self.context_length, self.context_length), dtype=torch.float16, device=TEST_DEVICE)  # [1, 1, context_length, context_length]
        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)  # [1] - position to extract logits from
        sample_update_mask = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16, device=TEST_DEVICE)  # [1, 1, context_length, 1]
        print("Sample inputs created (Fixed Window)")
        print(f"sample_input_ids shape: {sample_input_ids.shape}")
        print(f"sample_position_ids shape: {sample_position_ids.shape}")
        print(f"sample_causal_mask shape: {sample_causal_mask.shape}")
        print(f"sample_current_pos shape: {sample_current_pos.shape}")
        print(f"sample_update_mask shape: {sample_update_mask.shape}")

        print("Starting torch.jit.trace...")
        traced = torch.jit.trace(
            wrapper,
            (
                sample_input_ids,
                sample_position_ids,
                sample_causal_mask,
                sample_current_pos,
                sample_update_mask,
            ),
        )
        print("torch.jit.trace completed!")

        print("Starting CoreML conversion...")
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
                ct.TensorType(
                    name="update_mask", shape=sample_update_mask.shape, dtype=np.float16
                ),
            ],
            outputs=[
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
                ct.TensorType(name="logits3", dtype=np.float16),
                ct.TensorType(name="logits4", dtype=np.float16),
                ct.TensorType(name="logits5", dtype=np.float16),
                ct.TensorType(name="logits6", dtype=np.float16),
                ct.TensorType(name="logits7", dtype=np.float16),
                ct.TensorType(name="logits8", dtype=np.float16),
                ct.TensorType(name="logits9", dtype=np.float16),
                ct.TensorType(name="logits10", dtype=np.float16),
                ct.TensorType(name="logits11", dtype=np.float16),
                ct.TensorType(name="logits12", dtype=np.float16),
                ct.TensorType(name="logits13", dtype=np.float16),
                ct.TensorType(name="logits14", dtype=np.float16),
                ct.TensorType(name="logits15", dtype=np.float16),
                ct.TensorType(name="logits16", dtype=np.float16),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print("CoreML conversion completed!")

        return mlmodel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the converter."""

    parser = argparse.ArgumentParser(description="Convert Qwen model to CoreML format")

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model directory (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="qwen",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prefill",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Maximum context length",
    )
    parser.add_argument(
        "--lut",
        type=int,
        default=None,
        help="Use LUT quantization with N bits",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Split FFN/prefill into N chunks",
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["all"],
        default="all",
        help="Parts to convert (only 'all' supported)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for converted models",
    )

    return parser.parse_args()


def test_conversion(
    model: Optional[QwenForCausalLM] = None,
    model_path: Optional[str] = None,
    prefix: str = "qwen",
    context_length: int = CONTEXT_LENGTH,
    lut_bits: Optional[int] = None,
    batch_size: int = 64,
    output_dir: str = ".",
) -> ct.models.MLModel:
    """Convert a Qwen model and save the result."""
    print(f"test_conversion called with model_path={model_path}, prefix={prefix}")

    if model is None:
        if model_path is None:
            raise ValueError("model_path must be provided if model is None")

        config_path = os.path.join(model_path, "config.json")
        print(f"Looking for config at: {config_path}")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")

        print("Loading config...")
        config = QwenConfig.from_json(config_path)
        print(f"Config loaded: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
        
        print("Creating model...")
        model = QwenForCausalLM(config, enable_coreml=True)
        print("Loading pretrained weights...")
        model.load_pretrained_weights(model_path)
        print("Model loaded successfully!")

    print("Creating converter...")
    converter = QwenConverter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=lut_bits,
    )

    print("Starting conversion...")
    mlmodel = converter.convert()
    print("Conversion completed!")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}.mlpackage")
    print(f"Saving model to: {out_path}")
    mlmodel.save(out_path)
    print("Model saved successfully!")
    return mlmodel


def main() -> None:
    print("Starting qwen_converter main()...")
    args = parse_args()
    print(f"Parsed args: {args}")

    model_path = args.model if args.model else "Qwen/Qwen3-0.6B"

    print(f"\nConverting model from: {model_path}")
    print(f"Output filename prefix: {args.prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    if args.lut:
        print(f"LUT quantization: {args.lut} bits")
    if args.chunk:
        print(f"Splitting into {args.chunk} chunks")
    print(f"Converting part(s): {args.part}")

    try:
        print("\nCalling test_conversion()...")
        result = test_conversion(
            model_path=model_path,
            prefix=args.prefix,
            context_length=args.context_length,
            lut_bits=args.lut,
            batch_size=args.batch_size,
            output_dir=args.output,
        )
        print(f"Conversion completed successfully! Result: {type(result)}")
    except Exception as e:  # pragma: no cover - CLI tool
        print(f"\nError during conversion: {str(e)}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main() 
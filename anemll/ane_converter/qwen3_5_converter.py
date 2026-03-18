"""Converter for Qwen 3.5 models.

Qwen3.5 hybrid architecture:
- 75% Gated DeltaNet (linear attention) + 25% full attention
- Three state types: delta_state, conv_state, kv_cache
- 16-way split LM head (248K vocab)
- Partial RoPE (25%) on full attention layers only
"""

from __future__ import annotations

import argparse
import os
import warnings
try:
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
except Exception:
    SklearnConvergenceWarning = None

if SklearnConvergenceWarning is not None:
    warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
warnings.filterwarnings("ignore", message="Number of distinct clusters .* smaller than n_clusters")
from typing import Optional, List

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto

from .environment import require_coreml


def _apply_coremltools_mlstate_patch():
    """Patch _handle_unused_inputs_func to skip state-typed inputs.

    PR: https://github.com/apple/coremltools/pull/2661
    Bug: mb.identity() crashes on state-typed inputs during chunked conversion.
    Fix: filter out state inputs before calling identity.
    """
    try:
        from coremltools.converters.mil.backend.nn.passes import handle_unused_inputs as hui_mod
        from coremltools.converters.mil.mil import types as mil_types
        from coremltools.converters.mil.mil import Block
        from coremltools.converters.mil.mil import Builder as mb

        def _patched(f):
            unused = [
                v for _, v in f.inputs.items()
                if len(v.child_ops) == 0 and not mil_types.is_state(v.sym_type)
            ]
            with f:
                for v in unused:
                    v_tmp = mb.identity(x=v, name=v.name + "_tmp")
                    Block._copy_scope_info(v, v_tmp)

        hui_mod._handle_unused_inputs_func = _patched
        print("[patch] coremltools handle_unused_inputs patched (skip state inputs)")
    except Exception as e:
        print(f"[patch] WARNING: Could not apply coremltools patch: {e}")
        print("[patch] Conversion may fail if chunks have unused state buffers")

from .base_converter import BaseConverter
from .metadata import AddMetadata, ModelPart
from ..models.qwen3_5_model import (
    Qwen35ForCausalLM,
    Qwen35Config,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
)


class Qwen35Converter(BaseConverter):
    """Handle conversion of Qwen 3.5 models to Core ML."""

    model_cls = Qwen35ForCausalLM

    def __init__(
        self,
        model: Qwen35ForCausalLM,
        context_length: int = CONTEXT_LENGTH,
        batch_size: int = 64,
        lut_bits: int | None = 4,
        per_channel: int = 8,
        num_chunks: int = 1,
        argmax_in_model: bool = False,
        lut_embeddings_bits=None,
        lut_embeddings_per_channel=8,
        lut_lmhead_bits=None,
        lut_lmhead_per_channel=8,
    ) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.head_dim = model.model.config.head_dim
        self.converted_model = None
        self.num_chunks = num_chunks
        self.argmax_in_model = argmax_in_model
        self.lut_embeddings_bits = lut_embeddings_bits
        self.lut_embeddings_per_channel = lut_embeddings_per_channel
        self.lut_lmhead_bits = lut_lmhead_bits
        self.lut_lmhead_per_channel = lut_lmhead_per_channel

    @staticmethod
    def GetMLStateTypes(wrapper, start_layer=None, end_layer=None):
        """Get per-layer ct.StateType list for MLState conversion.

        Each buffer gets its own StateType — required because ANE can only
        handle 1-2 slice updates per buffer. With separate buffers (each
        updated once), ANE works with 48+ buffers.

        When start_layer/end_layer are specified, only includes buffers for
        layers in that range (avoids unused state buffers causing errors).

        Args:
            wrapper: The wrapper module being traced. Buffer names must match
                     named_buffers() paths from the wrapper's perspective.
        """
        states = []
        total_bytes = 0

        if start_layer is not None:
            # Determine which buffer indices this chunk uses
            from anemll.models.qwen3_5_model import get_layer_state_mapping
            config = None
            for _, mod in wrapper.named_modules():
                if hasattr(mod, 'config'):
                    config = mod.config
                    break
            if config is None:
                # Fallback: include all
                start_layer = None

        for name, buf in wrapper.named_buffers():
            if "rotary" in name:
                continue

            # If chunking, filter to only this chunk's buffers
            if start_layer is not None and config is not None:
                include = False
                el = end_layer if end_layer is not None else config.num_hidden_layers
                for layer_idx in range(start_layer, el):
                    state_type, state_idx = get_layer_state_mapping(layer_idx, config.layer_types)
                    if state_type == 'linear':
                        if name.endswith(f"delta_{state_idx}") or name.endswith(f"conv_{state_idx}"):
                            include = True
                            break
                    else:
                        if name.endswith(f"kv_key_{state_idx}") or name.endswith(f"kv_val_{state_idx}"):
                            include = True
                            break
                if not include:
                    continue

            states.append(ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(buf.shape), dtype=np.float16),
                name=name,
            ))
            total_bytes += buf.numel() * 2

        print(f"MLState buffers ({len(states)} total):")
        print(f"  {total_bytes / 1024 / 1024:.1f} MB on-chip state")
        return states

    @staticmethod
    def _make_palettizer_config(nbits, per_channel, num_workers):
        """Build an OpPalettizerConfig for the given bit-width / granularity."""
        if per_channel <= 0:
            return cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=nbits,
                granularity="per_tensor",
                num_kmeans_workers=num_workers if num_workers is not None else 1,
            )
        return cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
            granularity="per_grouped_channel",
            group_size=per_channel,
            num_kmeans_workers=num_workers if num_workers is not None else 1,
        )

    def postprocess(self, num_workers=None):
        """Apply LUT quantization if configured."""
        if self.converted_model is not None and self.lut_bits is not None:
            use_per_tensor = self.per_channel <= 0
            if use_per_tensor:
                print(
                    f"Applying LUT quantization with {self.lut_bits} bits using PER-TENSOR granularity "
                    f"with {num_workers if num_workers else 1} worker(s)..."
                )
            else:
                print(
                    f"Applying LUT quantization with {self.lut_bits} bits and {self.per_channel} channels "
                    f"per group using {num_workers if num_workers else 1} worker(s)..."
                )
            try:
                with warnings.catch_warnings():
                    if SklearnConvergenceWarning is not None:
                        warnings.simplefilter('ignore', SklearnConvergenceWarning)
                    warnings.simplefilter('ignore', UserWarning)

                    global_cfg = self._make_palettizer_config(
                        self.lut_bits, self.per_channel, num_workers
                    )

                    op_name_configs = {}
                    has_overrides = (
                        self.lut_embeddings_bits is not None
                        or self.lut_lmhead_bits is not None
                    )

                    if has_overrides:
                        prog = self.converted_model._mil_program  # noqa: SLF001
                        for fn_name in prog.functions:
                            fn = prog.functions[fn_name]
                            for op in fn.operations:
                                op_name = op.name or ""
                                if self.lut_embeddings_bits is not None and "embed_tokens" in op_name:
                                    op_name_configs[op_name] = self._make_palettizer_config(
                                        self.lut_embeddings_bits,
                                        self.lut_embeddings_per_channel,
                                        num_workers,
                                    )
                                elif self.lut_lmhead_bits is not None and "lm_head" in op_name:
                                    op_name_configs[op_name] = self._make_palettizer_config(
                                        self.lut_lmhead_bits,
                                        self.lut_lmhead_per_channel,
                                        num_workers,
                                    )

                        if op_name_configs:
                            embed_count = sum(1 for k in op_name_configs if "embed_tokens" in k)
                            lmhead_count = sum(1 for k in op_name_configs if "lm_head" in k)
                            if self.lut_embeddings_bits is not None:
                                print(f"  Embeddings: {self.lut_embeddings_bits}-bit LUT "
                                      f"(per_channel={self.lut_embeddings_per_channel}) -> {embed_count} ops")
                            if self.lut_lmhead_bits is not None:
                                print(f"  LM head: {self.lut_lmhead_bits}-bit LUT "
                                      f"(per_channel={self.lut_lmhead_per_channel}) -> {lmhead_count} ops")

                    config = cto.coreml.OptimizationConfig(
                        global_config=global_cfg,
                        op_name_configs=op_name_configs if op_name_configs else None,
                    )

                    try:
                        self.converted_model = cto.coreml.palettize_weights(
                            self.converted_model, config
                        )
                    except Exception as e:
                        print(f"Warning: palettize_weights raised: {e}")
                        print("Retrying without per-op overrides...")
                        fallback_config = cto.coreml.OptimizationConfig(
                            global_config=global_cfg,
                        )
                        self.converted_model = cto.coreml.palettize_weights(
                            self.converted_model, fallback_config
                        )
                print("LUT quantization completed successfully")

            except Exception as e:
                print(f"LUT quantization failed: {str(e)}")
                print("Continuing without quantization...")

    @staticmethod
    def _reset_state_buffers(module: torch.nn.Module) -> None:
        """Clear all mutable state buffers to avoid trace side-effects."""
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                if any(s in name for s in ("delta_", "conv_", "kv_key_", "kv_val_")):
                    buffer.zero_()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(
        self, part: str = "full"
    ) -> ct.models.MLModel | List[ct.models.MLModel]:
        """Convert the wrapped model to CoreML format."""
        print(f"Qwen35Converter.convert() called with part={part}")
        require_coreml()
        print("Calling preprocess()...")
        self.preprocess()

        if part in ("full", "all", "123"):
            print("Converting full model...")
            mlmodel = self.convert_to_coreml(self.model)
        elif part == "monolithic":
            print("Converting monolithic model...")
            mlmodel = self.convert_monolithic(
                self.model, is_prefill=False, argmax_in_model=self.argmax_in_model,
            )
        elif part == "monolithic_prefill":
            print("Converting monolithic prefill model...")
            mlmodel = self.convert_monolithic(
                self.model, is_prefill=True, argmax_in_model=False,
            )
        elif part in ("embeddings", "1"):
            print("Converting embeddings...")
            mlmodel = self.convert_part_1(self.model)
        elif part in ("prefill", "2_prefill"):
            print("Converting prefill...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_prefill(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_prefill(self.model)
        elif part == "2":
            print("Converting FFN...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2(self.model)
        elif part == "3":
            print("Converting LM head...")
            mlmodel = self.convert_part_3(
                self.model, argmax_in_model=self.argmax_in_model,
            )
        else:
            raise ValueError(f"Unsupported part: {part}")

        print("Calling postprocess()...")
        self.postprocess()
        print("Qwen35Converter.convert() completed")
        return mlmodel

    def convert_to_coreml(self, model: Qwen35ForCausalLM) -> ct.models.MLModel:
        """Convert the entire model to CoreML."""
        require_coreml()

        class Wrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM, context_length: int) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length

            def forward(self, input_ids, position_ids, causal_mask, current_pos,
                        update_mask):
                # MLState: model accesses its own per-layer buffers directly.
                logits = self.model(
                    input_ids=input_ids,
                    update_mask=update_mask,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    IN_PREFILL=False,
                )

                return logits

        wrapper = Wrapper(model, self.context_length)
        wrapper.eval()

        sample_input_ids = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
        sample_position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        sample_causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )
        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        sample_update_mask = torch.zeros(
            (1, 1, self.context_length, 1), dtype=torch.float16, device=TEST_DEVICE
        )

        self._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper,
            (sample_input_ids, sample_position_ids, sample_causal_mask,
             sample_current_pos, sample_update_mask),
        )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=np.int32),
                ct.TensorType(name="position_ids", shape=sample_position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=sample_current_pos.shape, dtype=np.int32),
                ct.TensorType(name="update_mask", shape=sample_update_mask.shape, dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 17)
            ],
            states=self.GetMLStateTypes(wrapper, start_layer=None, end_layer=None),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model

        return mlmodel

    # --------------------------------------------------------------
    # Part-based conversion helpers
    # --------------------------------------------------------------
    def convert_part_1(self, model: Qwen35ForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer only."""
        require_coreml()
        return self.convert_embeddings(model)

    def convert_part_3(
        self, model: Qwen35ForCausalLM, argmax_in_model: bool = False
    ) -> ct.models.MLModel:
        """Convert LM head only."""
        require_coreml()

        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM, argmax_mode: bool = False) -> None:
                super().__init__()
                self.argmax_mode = argmax_mode
                if hasattr(model, "lm_head16_1"):
                    self.heads = [getattr(model, f"lm_head16_{i}") for i in range(1, 17)]
                    self.mode = "16"
                elif hasattr(model, "lm_head8_1"):
                    self.heads = [getattr(model, f"lm_head8_{i}") for i in range(1, 9)]
                    self.mode = "8"
                elif hasattr(model, "lm_head2_1"):
                    self.heads = [model.lm_head2_1, model.lm_head2_2]
                    self.mode = "2"
                elif hasattr(model, "lm_head1"):
                    self.head = model.lm_head1
                    self.mode = "1"
                else:
                    self.head = model.lm_head
                    self.mode = "linear"

            def forward(self, hidden_states: torch.Tensor):
                if self.mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.mode in ("16", "8", "2"):
                    logits_list = [
                        h(hidden_states).squeeze(2).transpose(1, 2) for h in self.heads
                    ]
                elif self.mode == "1":
                    logits_list = [self.head(hidden_states).squeeze(2).transpose(1, 2)]
                else:
                    logits_list = [self.head(hidden_states)]

                if self.argmax_mode:
                    all_idx = []
                    all_val = []
                    for logits in logits_list:
                        chunk_argmax = torch.argmax(logits, dim=-1, keepdim=True)
                        chunk_max_val = torch.gather(logits, -1, chunk_argmax)
                        all_idx.append(chunk_argmax.to(torch.int32))
                        all_val.append(chunk_max_val)
                    argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)
                    argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)
                    return (argmax_idx, argmax_val)

                if self.mode == "16":
                    return tuple(logits_list)
                if self.mode == "8":
                    return tuple(logits_list)
                if self.mode == "2":
                    return logits_list[0], logits_list[1]
                return logits_list[0]

        wrapper = LMHeadWrapper(model, argmax_mode=argmax_in_model)
        wrapper.eval()

        for param in wrapper.parameters():
            param.requires_grad = False

        sample_input = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, sample_input)

        if argmax_in_model:
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
        elif getattr(wrapper, "mode") == "16":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 17)
            ]
        elif getattr(wrapper, "mode") == "8":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 9)
            ]
        elif getattr(wrapper, "mode") == "2":
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states", shape=sample_input.shape, dtype=np.float16
                )
            ],
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=None)
            mlmodel = self.converted_model

        return mlmodel

    def convert_part_2(
        self, model: Qwen35ForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert transformer layers for generation (FFN decode)."""
        require_coreml()
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            base, rem = divmod(total_layers, total_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
        else:
            start_layer = 0
            end_layer = None

        class FFNWrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM, start_layer: int, end_layer: int) -> None:
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer

            def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                        update_mask):
                # MLState: model accesses its own per-layer buffers directly.
                # No state input/output — state lives on ANE chip.
                rotary = self.model.model.get_rotary_embeddings_s(current_pos)
                out = self.model.model.process_layers(
                    hidden_states,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=False,
                    update_mask=update_mask,
                )
                # Apply final norm only on the last chunk
                if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                    out = self.model.model.norm(out)

                return out

        wrapper = FFNWrapper(model, start_layer, end_layer)
        wrapper.eval()

        hidden_states = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE
        )
        position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        update_mask = torch.zeros(
            (1, 1, self.context_length, 1), dtype=torch.float16, device=TEST_DEVICE
        )

        self._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos,
                      update_mask)
        )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
                ct.TensorType(name="update_mask", shape=update_mask.shape, dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ],
            states=self.GetMLStateTypes(wrapper, start_layer, end_layer),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            num_workers = None if total_chunks > 1 else 8
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model

        return mlmodel

    def convert_part_2_prefill(
        self, model: Qwen35ForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert transformer layers for prefill mode."""
        require_coreml()
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            base, rem = divmod(total_layers, total_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
        else:
            start_layer = 0
            end_layer = None

        class PrefillWrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM, start_layer=0, end_layer=None):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                # MLState: model accesses its own per-layer buffers directly.
                rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
                out = self.model.model.process_layers(
                    hidden_states,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=True,
                )

                # Skip normalization for prefill - only state update matters
                if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                    out = out[:, 0:1, :]

                return out

        wrapper = PrefillWrapper(model, start_layer, end_layer)
        wrapper.eval()

        hidden_states = torch.zeros(
            (1, self.batch_size, model.config.hidden_size),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )
        position_ids = torch.zeros(
            (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
        )
        causal_mask = torch.zeros(
            (1, 1, self.batch_size, self.context_length),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        self._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos)
        )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ],
            states=self.GetMLStateTypes(wrapper, start_layer, end_layer),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            num_workers = None if total_chunks > 1 else 8
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model

        return mlmodel

    def convert_embeddings(self, model: Qwen35ForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer to CoreML format."""
        require_coreml()
        print("\nConverting Qwen3.5 embeddings layer...")

        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.embed_tokens = model.model.embed_tokens

            def forward(self, input_ids):
                hidden_states = self.embed_tokens(input_ids)
                return hidden_states.to(MODEL_DTYPE)

        wrapper = EmbeddingsWrapper(model)
        wrapper.eval()

        sample_input = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)

        traced_model = torch.jit.trace(wrapper, sample_input)

        input_shape = ct.EnumeratedShapes(
            shapes=[[1, 1], [1, self.batch_size]],
            default=[1, 1],
        )

        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)
            ],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=None)
            mlmodel = self.converted_model

        return mlmodel

    def convert_monolithic(
        self,
        model: Qwen35ForCausalLM,
        is_prefill: bool = False,
        argmax_in_model: bool = False,
    ) -> ct.models.MLModel:
        """Convert full model (embeddings + FFN + LM head) to single CoreML model."""
        require_coreml()
        mode_str = "prefill" if is_prefill else "inference"
        print(f"\nConverting monolithic model for {mode_str} mode...")

        class MonolithicWrapper(torch.nn.Module):
            def __init__(
                self,
                model: Qwen35ForCausalLM,
                context_length: int,
                is_prefill: bool,
                argmax_in_model: bool = False,
            ) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length
                self.is_prefill = is_prefill
                self.argmax_in_model = argmax_in_model

                if hasattr(model, "lm_head16_1"):
                    self.lm_head_mode = "16"
                    self.lm_heads = [
                        getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                    ]
                elif hasattr(model, "lm_head8_1"):
                    self.lm_head_mode = "8"
                    self.lm_heads = [
                        getattr(model, f"lm_head8_{i}") for i in range(1, 9)
                    ]
                elif hasattr(model, "lm_head2_1"):
                    self.lm_head_mode = "2"
                    self.lm_heads = [model.lm_head2_1, model.lm_head2_2]
                elif hasattr(model, "lm_head1"):
                    self.lm_head_mode = "1"
                    self.lm_head = model.lm_head1
                else:
                    self.lm_head_mode = "linear"
                    self.lm_head = model.lm_head

            def forward(self, input_ids, position_ids, causal_mask, current_pos,
                        update_mask):
                # MLState: model accesses its own per-layer buffers directly.

                # Step 1: Embeddings
                hidden_states = self.model.model.embed_tokens(input_ids)
                hidden_states = hidden_states.to(MODEL_DTYPE)

                # Step 2: Transformer layers
                if self.is_prefill:
                    rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
                else:
                    rotary = self.model.model.get_rotary_embeddings_s(current_pos)

                hidden_states = self.model.model.process_layers(
                    hidden_states,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=0,
                    end_layer=None,
                    IN_PREFILL=self.is_prefill,
                    update_mask=update_mask if not self.is_prefill else None,
                )

                hidden_states = self.model.model.norm(hidden_states)

                # Step 3: LM Head
                if self.lm_head_mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.lm_head_mode in ("16", "8", "2"):
                    logits_list = [
                        h(hidden_states).squeeze(2).transpose(1, 2)
                        for h in self.lm_heads
                    ]
                elif self.lm_head_mode == "1":
                    logits_list = [self.lm_head(hidden_states).squeeze(2).transpose(1, 2)]
                else:
                    logits_list = [self.lm_head(hidden_states)]

                if self.argmax_in_model and not self.is_prefill:
                    all_idx = []
                    all_val = []
                    for logits in logits_list:
                        chunk_argmax = torch.argmax(logits, dim=-1, keepdim=True)
                        chunk_max_val = torch.gather(logits, -1, chunk_argmax)
                        all_idx.append(chunk_argmax.to(torch.int32))
                        all_val.append(chunk_max_val)
                    argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)
                    argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)
                    return (argmax_idx, argmax_val)

                return tuple(logits_list)

        wrapper = MonolithicWrapper(
            model, self.context_length, is_prefill, argmax_in_model=argmax_in_model,
        )
        wrapper.eval()

        for param in wrapper.parameters():
            param.requires_grad = False

        argmax_str = ", argmax_in_model=True" if (argmax_in_model and not is_prefill) else ""
        print(f"Monolithic wrapper created (LM head mode: {wrapper.lm_head_mode}{argmax_str})")

        if is_prefill:
            sample_input_ids = torch.zeros(
                (1, self.batch_size), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, self.batch_size, self.context_length),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )
        else:
            sample_input_ids = torch.zeros(
                (1, 1), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (1,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, 1, self.context_length),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )

        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        sample_update_mask = torch.zeros(
            (1, 1, self.context_length, 1), dtype=torch.float16, device=TEST_DEVICE
        )

        self._reset_state_buffers(wrapper)
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (sample_input_ids, sample_position_ids, sample_causal_mask,
                 sample_current_pos, sample_update_mask),
            )

        if argmax_in_model and not is_prefill:
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
        elif wrapper.lm_head_mode == "16":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16)
                for i in range(1, 17)
            ]
        elif wrapper.lm_head_mode == "8":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16)
                for i in range(1, 9)
            ]
        elif wrapper.lm_head_mode == "2":
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=np.int32),
                ct.TensorType(name="position_ids", shape=sample_position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=sample_current_pos.shape, dtype=np.int32),
                ct.TensorType(name="update_mask", shape=sample_update_mask.shape, dtype=np.float16),
            ],
            outputs=outputs,
            states=self.GetMLStateTypes(wrapper, start_layer=None, end_layer=None),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model

        return mlmodel


def parse_lut_arg(lut_value):
    """Parse LUT argument that can be 'bits', 'bits,per_channel', or 'bits,0' for per-tensor."""
    if lut_value is None:
        return None, 8

    if isinstance(lut_value, int):
        return lut_value, 8

    lut_str = str(lut_value).strip().lower()

    if lut_str in ('none', 'no', 'false', ''):
        return None, 8

    if ',' in lut_str:
        parts = lut_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid LUT format: {lut_value}. Expected 'bits' or 'bits,per_channel'")
        try:
            lut_bits = int(parts[0])
            per_channel_str = parts[1].strip().lower()
            if per_channel_str in ('tensor', 't', '0'):
                per_channel = 0
            else:
                per_channel = int(parts[1])
            return lut_bits, per_channel
        except ValueError:
            raise ValueError(f"Invalid LUT format: {lut_value}. Expected 'bits' or 'bits,per_channel'")
    else:
        try:
            lut_bits = int(lut_str)
            return lut_bits, 8
        except ValueError:
            raise ValueError(f"Invalid LUT bits value: {lut_value}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the converter."""
    parser = argparse.ArgumentParser(description="Convert Qwen 3.5 model to CoreML format")

    parser.add_argument(
        "--model", type=str,
        help="Path to model directory (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--prefix", type=str, default="qwen35",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for prefill",
    )
    parser.add_argument(
        "--context-length", type=int, default=CONTEXT_LENGTH,
        help="Maximum context length",
    )
    parser.add_argument(
        "--lut", type=str, default=None,
        help='Use LUT quantization with N bits, optionally "bits,per_channel" (e.g., "6,4"). Default per_channel is 8',
    )
    parser.add_argument(
        "--chunk", type=int, default=None,
        help="Split FFN/prefill into N chunks",
    )
    parser.add_argument(
        "--part", type=str,
        choices=["1", "2", "2_prefill", "3", "all", "full", "prefill", "embeddings",
                 "monolithic", "monolithic_prefill"],
        default="all",
        help="Model part to convert (monolithic = single file with embed+FFN+lmhead)",
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Output directory for converted models",
    )
    parser.add_argument(
        "--argmax", action="store_true",
        help="Compute argmax inside LM head for part 3 / monolithic inference.",
    )
    parser.add_argument(
        '--lut-embeddings', type=str, default=None,
        help="Override LUT for embeddings in monolithic models. Same format as --lut.",
    )
    parser.add_argument(
        '--lut-lmhead', type=str, default=None,
        help="Override LUT for LM head in monolithic models. Same format as --lut.",
    )

    return parser.parse_args()


def test_conversion(
    model: Optional[Qwen35ForCausalLM] = None,
    model_path: Optional[str] = None,
    prefix: str = "qwen35",
    context_length: int = CONTEXT_LENGTH,
    lut_bits: Optional[int] = None,
    batch_size: int = 64,
    output_dir: str = ".",
    part: str = "full",
    num_chunks: int = 1,
    per_channel: int = 8,
    argmax_in_model: bool = False,
    lut_embeddings_bits=None,
    lut_embeddings_per_channel=8,
    lut_lmhead_bits=None,
    lut_lmhead_per_channel=8,
) -> ct.models.MLModel | List[ct.models.MLModel]:
    """Convert a Qwen3.5 model and save the result."""
    _apply_coremltools_mlstate_patch()
    print(
        f"test_conversion called with model_path={model_path}, prefix={prefix}, part={part}"
    )

    if model is None:
        if model_path is None:
            raise ValueError("model_path must be provided if model is None")

        config_path = os.path.join(model_path, "config.json")
        print(f"Looking for config at: {config_path}")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")

        print("Loading config...")
        config = Qwen35Config.from_json(config_path)
        print(
            f"Config loaded: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}"
        )

        config.context_length = context_length
        config.state_length = max(config.state_length, context_length)
        print(
            f"Updated config: context_length={config.context_length}, state_length={config.state_length}"
        )

        print("Creating model...")
        model = Qwen35ForCausalLM(config, enable_coreml=True)
        print("Loading pretrained weights...")
        model.load_pretrained_weights(model_path)
        print("Model loaded successfully!")

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("Model set to eval mode and gradients disabled")

    print("Creating converter...")
    converter = Qwen35Converter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=lut_bits,
        num_chunks=num_chunks,
        per_channel=per_channel,
        argmax_in_model=argmax_in_model,
        lut_embeddings_bits=lut_embeddings_bits,
        lut_embeddings_per_channel=lut_embeddings_per_channel,
        lut_lmhead_bits=lut_lmhead_bits,
        lut_lmhead_per_channel=lut_lmhead_per_channel,
    )

    print("Starting conversion...")
    mlmodel = converter.convert(part=part)
    print("Conversion completed!")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(mlmodel, list):
        models = mlmodel
    else:
        models = [mlmodel]

    vocab_size = int(getattr(model.config, "vocab_size", 0)) if model is not None else None
    lm_head_chunk_sizes = None
    if part in ["3", "monolithic"]:
        if hasattr(model, "lm_head16_1"):
            lm_head_chunk_sizes = [
                int(getattr(model, f"lm_head16_{i}").out_channels) for i in range(1, 17)
            ]
        elif hasattr(model, "lm_head8_1"):
            lm_head_chunk_sizes = [
                int(getattr(model, f"lm_head8_{i}").out_channels) for i in range(1, 9)
            ]
        elif hasattr(model, "lm_head2_1"):
            lm_head_chunk_sizes = [
                int(model.lm_head2_1.out_channels),
                int(model.lm_head2_2.out_channels),
            ]
        elif hasattr(model, "lm_head1"):
            lm_head_chunk_sizes = [int(model.lm_head1.out_channels)]
        elif hasattr(model, "lm_head"):
            lm_head_chunk_sizes = [int(model.lm_head.out_channels)]

    for i, m in enumerate(models):
        AddMetadata(
            m,
            {
                "context_length": context_length,
                "batch_size": batch_size if part in ["2_prefill", "prefill", "monolithic_prefill"] else None,
                "lut_bits": lut_bits,
                "num_chunks": num_chunks if part in ["2", "2_prefill"] else None,
                "chunk_no": i + 1 if part in ["2", "2_prefill"] else None,
                "split_part": (
                    ModelPart.FULL.value if part in ["full", "all", "123"] else part
                ),
                "argmax_in_model": argmax_in_model if part in ["3", "monolithic"] else None,
                "vocab_size": vocab_size if part in ["3", "monolithic"] else None,
                "lm_head_chunk_sizes": lm_head_chunk_sizes if part in ["3", "monolithic"] else None,
            },
        )
        fname = f"{prefix}"
        if part in ["1", "embeddings"]:
            fname += "_embeddings"
        elif part in ["3"]:
            fname += "_lm_head"
        elif part == "monolithic":
            fname += "_monolithic"
        elif part == "monolithic_prefill":
            fname += "_monolithic_prefill"
        elif part in ["2", "2_prefill"]:
            base = "FFN" if part == "2" else "prefill"
            fname += f"_{base}"
            if lut_bits is not None:
                fname += f"_lut{lut_bits}"
            fname += f"_chunk_{i+1:02d}of{num_chunks:02d}"
        if part in ["full", "all", "123"]:
            fname += ""
        if part not in ["2", "2_prefill"]:
            if lut_bits is not None:
                fname += f"_lut{lut_bits}"
            fname += ".mlpackage"
        else:
            fname += ".mlpackage"
        out_path = os.path.join(output_dir, fname)
        print(f"Saving model to: {out_path}")
        m.save(out_path)

    return mlmodel


def main() -> None:
    print("Starting qwen3_5_converter main()...")
    args = parse_args()
    print(f"Parsed args: {args}")

    lut_bits, per_channel = parse_lut_arg(args.lut)
    lut_embeddings_bits, lut_embeddings_per_channel = parse_lut_arg(args.lut_embeddings)
    lut_lmhead_bits, lut_lmhead_per_channel = parse_lut_arg(args.lut_lmhead)

    model_path = args.model if args.model else "Qwen/Qwen3.5-0.8B"

    print(f"\nConverting model from: {model_path}")
    print(f"Output filename prefix: {args.prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    if lut_bits:
        print(f"LUT quantization: {lut_bits} bits, per_channel group size: {per_channel}")
    if lut_embeddings_bits is not None:
        print(f"LUT embeddings override: {lut_embeddings_bits} bits, per_channel={lut_embeddings_per_channel}")
    if lut_lmhead_bits is not None:
        print(f"LUT lm_head override: {lut_lmhead_bits} bits, per_channel={lut_lmhead_per_channel}")
    if args.chunk:
        print(f"Splitting into {args.chunk} chunks")
    if args.argmax:
        print("Argmax in model: enabled")
    print(f"Converting part(s): {args.part}")

    part_map = {"full": "all", "embeddings": "1", "prefill": "2_prefill"}
    part = part_map.get(args.part, args.part)

    try:
        print("\nCalling test_conversion()...")
        result = test_conversion(
            model_path=model_path,
            prefix=args.prefix,
            context_length=args.context_length,
            lut_bits=lut_bits,
            batch_size=args.batch_size,
            output_dir=args.output,
            part=part,
            num_chunks=args.chunk or 1,
            per_channel=per_channel,
            argmax_in_model=args.argmax,
            lut_embeddings_bits=lut_embeddings_bits,
            lut_embeddings_per_channel=lut_embeddings_per_channel,
            lut_lmhead_bits=lut_lmhead_bits,
            lut_lmhead_per_channel=lut_lmhead_per_channel,
        )
        print(f"Conversion completed successfully! Result: {type(result)}")
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

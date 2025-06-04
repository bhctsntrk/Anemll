"""Qwen 3 model implementation for ANEMLL.

This module provides a lightweight implementation of the Qwen 3 architecture
adapted to the Apple Neural Engine restrictions.  All dense layers are expressed
as ``nn.Conv2d`` with ``kernel_size=1`` and weights are loaded from Hugging Face
checkpoints with the correct reshaping.  Only the pieces required for the unit
 tests are implemented.
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Qwen 3 model implementation adapted from llama_model.py
# ---------------------------------------------------------------------------

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 512


class QwenConfig:
    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["QwenForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 128000)
        self.eos_token_id = kwargs.get("eos_token_id", 128001)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 14336)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.model_type = kwargs.get("model_type", "qwen3")
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.head_dim = kwargs.get(
            "head_dim",
            self.hidden_size // max(1, self.num_attention_heads),
        )
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "qwen3")
        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.torch_required = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.40.0.dev0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 128257)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", CONTEXT_LENGTH)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# -----------------------------------------------------------------------------
# Qwen building blocks
# -----------------------------------------------------------------------------


class QwenRMSNorm(nn.Module):
    """RMSNorm used in Qwen models."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            hidden_states, self.weight.shape, self.weight, eps=self.eps
        ).to(MODEL_DTYPE)


class QwenHeadNorm(nn.Module):
    """Per-head RMSNorm for query and key projections."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.weight.shape[0],), self.weight, eps=self.eps).to(
            MODEL_DTYPE
        )


class QwenRotaryEmbedding(nn.Module):
    """Simple rotary positional embedding."""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0)
        self.sin_cached = emb.sin().unsqueeze(0)

    def forward(self, x: torch.Tensor, seq_len: int | None = None):
        return (
            self.cos_cached[:, : seq_len or x.shape[1]].to(x.dtype),
            self.sin_cached[:, : seq_len or x.shape[1]].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(bsz, n_kv * n_rep, seq_len, head_dim)


class QwenMLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Conv2d(
            self.hidden_size, self.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.up_proj = nn.Conv2d(
            self.hidden_size, self.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.down_proj = nn.Conv2d(
            self.intermediate_size, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
        gate = self.act_fn(self.gate_proj(hs)) * self.up_proj(hs)
        hidden_states = self.down_proj(gate)
        return hidden_states.squeeze(2).permute(0, 2, 1)


class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.rotary_emb = QwenRotaryEmbedding(config)

        self.q_proj = nn.Conv2d(
            self.hidden_size,
            self.num_heads * self.head_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.o_proj = nn.Conv2d(
            self.num_heads * self.head_dim,
            self.hidden_size,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.q_norm = QwenHeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QwenHeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
        query_states = (
            self.q_proj(hs)
            .view(bsz, self.num_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )
        key_states = (
            self.k_proj(hs)
            .view(bsz, self.num_kv_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )
        value_states = (
            self.v_proj(hs)
            .view(bsz, self.num_kv_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )

        n_rep = self.num_heads // self.num_kv_heads
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = self.rotary_emb(hidden_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        )
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)
        )
        out = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return out.squeeze(2).permute(0, 2, 1)


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, causal_mask, position_ids, current_pos
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(
            TEST_DEVICE
        )
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, position_ids, current_pos)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_pretrained_weights(self, model_path: str) -> bool:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )

        conv_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.", "") if k.startswith("model.") else k
            if "lm_head.weight" in new_k:
                continue
            if any(
                proj in new_k
                for proj in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                ]
            ):
                conv_state[new_k] = v.view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[new_k] = v

        missing, unexpected = self.load_state_dict(conv_state, strict=False)
        missing = [m for m in missing if "rotary_emb.inv_freq" not in m]
        if missing or unexpected:
            print("Missing keys", missing)
            print("Unexpected keys", unexpected)
        return not missing and not unexpected


class QwenForCausalLM(nn.Module):
    config_class = QwenConfig

    def __init__(self, config: QwenConfig, **kwargs) -> None:
        super().__init__()
        self.model = QwenModel(config)
        self.lm_head = nn.Conv2d(
            config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)

    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, causal_mask, position_ids, current_pos)
        logits = self.lm_head(hidden_states.permute(0, 2, 1).unsqueeze(2))
        return logits.squeeze(2).permute(0, 2, 1)

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not self.model.load_pretrained_weights(model_path):
            return False
        path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(path):
            state = safetensors.torch.load_file(path)
            w = state.get("lm_head.weight")
            if w is not None:
                self.lm_head.weight.data.copy_(w.view(w.shape[0], w.shape[1], 1, 1))
        return True

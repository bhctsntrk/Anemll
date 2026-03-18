"""Qwen 3.5 model implementation for ANEMLL.

Hybrid architecture: 75% Gated DeltaNet (linear attention) + 25% full attention.
This is the first open-source implementation of Gated DeltaNet on Apple Neural Engine.

Key differences from Qwen 3:
- Gated DeltaNet recurrent layers (linear_attention) with fixed-size state
- Full attention layers with sigmoid output gate and partial RoPE (25%)
- Three state types: delta_state, conv_state, kv_cache
- CausalConv1D before DeltaNet split
- GatedRMSNorm on DeltaNet output
- Q/K head normalization on full attention layers
"""
from __future__ import annotations

import os
import json
import math
from typing import Dict, List, Tuple

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 256

# Cache/state configuration
ENABLE_SPLIT_CACHE = True  # Use per-layer state buffers for MLState compatibility
STATE_LENGTH = CONTEXT_LENGTH
CONV_STATE_PAD = 32  # Pad conv_state last axis to 32 for ANE alignment
DISABLE_KV_CACHE = False

# LM head configuration (following qwen_model.py pattern)
ENABLE_CONV2D = bool(1)
ENABLE_VACAB_SPLIT = bool(1)
ENABLE_VACAB_SPLIT8 = bool(0)
ENABLE_VACAB_SPLIT16 = bool(1)
ENABLE_LOGITS2 = bool(1)
ENABLE_COREML = bool(0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Qwen35Config:
    def __init__(self, **kwargs):
        # Handle text_config nesting (multimodal config structure)
        if "text_config" in kwargs and "hidden_size" not in kwargs:
            text_cfg = dict(kwargs["text_config"])
            # Carry over top-level fields
            for key in ("tie_word_embeddings", "architectures", "transformers_version"):
                if key in kwargs and key not in text_cfg:
                    text_cfg[key] = kwargs[key]
            # Also extract rope_parameters if nested
            if "rope_parameters" in text_cfg:
                rp = text_cfg["rope_parameters"]
                if "partial_rotary_factor" not in text_cfg:
                    text_cfg["partial_rotary_factor"] = rp.get("partial_rotary_factor", 0.25)
                if "rope_theta" not in text_cfg:
                    text_cfg["rope_theta"] = rp.get("rope_theta", 10000000.0)
            kwargs = text_cfg

        self.architectures = kwargs.get("architectures", ["Qwen3_5ForConditionalGeneration"])
        self.model_type = kwargs.get("model_type", "qwen3_5_text")
        self.hidden_size = kwargs.get("hidden_size", 1024)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.intermediate_size = kwargs.get("intermediate_size", 3584)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.vocab_size = kwargs.get("vocab_size", 248320)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.hidden_act = kwargs.get("hidden_act", "silu")

        # Full attention config
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.head_dim = kwargs.get("head_dim", 256)
        self.attn_output_gate = kwargs.get("attn_output_gate", True)
        self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 0.25)
        self.rope_theta = kwargs.get("rope_theta", 10000000.0)

        # Linear attention (DeltaNet) config
        self.linear_num_key_heads = kwargs.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = kwargs.get("linear_num_value_heads", 16)
        self.linear_key_head_dim = kwargs.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = kwargs.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = kwargs.get("linear_conv_kernel_dim", 4)

        # Layer types
        if "layer_types" in kwargs:
            self.layer_types = kwargs["layer_types"]
        else:
            # Default: 3 linear + 1 full, repeated
            interval = kwargs.get("full_attention_interval", 4)
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if (i + 1) % interval == 0:
                    self.layer_types.append("full_attention")
                else:
                    self.layer_types.append("linear_attention")

        # Context / cache
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", self.context_length)
        self.batch_size = kwargs.get("batch_size", 64)

        # Derived counts
        self.num_linear_layers = sum(1 for t in self.layer_types if t == "linear_attention")
        self.num_full_layers = sum(1 for t in self.layer_types if t == "full_attention")

        # DeltaNet QKV dimension
        self.linear_qkv_dim = (
            self.linear_num_key_heads * self.linear_key_head_dim +  # Q
            self.linear_num_key_heads * self.linear_key_head_dim +  # K
            self.linear_num_value_heads * self.linear_value_head_dim  # V
        )

        # Tokenizer
        self.eos_token_id = kwargs.get("eos_token_id", 248044)
        self.use_cache = kwargs.get("use_cache", True)

        # Force rotation mode (for CoreML tracing)
        self.force_rotation_mode = kwargs.get("force_rotation_mode", None)

    def get_linear_layer_indices(self):
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    def get_full_layer_indices(self):
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def get_layer_state_mapping(layer_idx, layer_types):
    """Map layer index to state type and index within that state.

    Returns:
        ('linear', linear_idx) for DeltaNet layers
        ('full', full_idx) for full attention layers
    """
    if layer_types[layer_idx] == "linear_attention":
        idx = sum(1 for i in range(layer_idx) if layer_types[i] == "linear_attention")
        return 'linear', idx
    else:
        idx = sum(1 for i in range(layer_idx) if layer_types[i] == "full_attention")
        return 'full', idx


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Qwen35RMSNorm(nn.Module):
    """ANE-optimized RMSNorm using doubled-concat trick.

    Qwen3.5 uses (1 + weight) scaling like Gemma, NOT plain weight like Qwen3.
    Weight is initialized to zeros, forward applies (1 + weight).
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = x.shape[-1]
        normed = F.layer_norm(
            doubled, (2 * hidden_size,), None, None, float(self.variance_epsilon)
        )
        normed = normed[..., :hidden_size]
        # Qwen3.5 style: (1 + weight) scaling
        return normed * (1.0 + self.weight.to(normed.dtype, copy=False).to(normed.device, copy=False))


class Qwen35HeadNorm(nn.Module):
    """Per-head RMSNorm for Q/K normalization. Uses (1+weight) like Qwen35RMSNorm."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = x.shape[-1]
        normed = F.layer_norm(
            doubled, (2 * hidden_size,), None, None, float(self.variance_epsilon)
        )
        normed = normed[..., :hidden_size]
        return normed * (1.0 + self.weight.to(normed.dtype, copy=False).to(normed.device, copy=False))


class Qwen35GatedRMSNorm(nn.Module):
    """GatedRMSNorm = RMSNorm(x) * SiLU(z) — used after DeltaNet output."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, z):
        """x: attention output, z: gate input. Both [B, S, hidden]."""
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = x.shape[-1]
        normed = F.layer_norm(
            doubled, (2 * hidden_size,), None, None, float(self.variance_epsilon)
        )
        normed = normed[..., :hidden_size]
        normed = normed * self.weight.to(normed.dtype, copy=False).to(normed.device, copy=False)
        return normed * F.silu(z)


class Qwen35RotaryEmbedding(nn.Module):
    """Partial RoPE: only applies to first partial_rotary_factor of head_dim."""

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_dim = int(config.head_dim * config.partial_rotary_factor)  # 64

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        max_pos = max(config.context_length, config.state_length) * 2
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, rope_dim]
        self.cos_cached = emb.cos().unsqueeze(0)  # [1, max_pos, rope_dim]
        self.sin_cached = emb.sin().unsqueeze(0)

    def forward(self, position_ids):
        """Returns cos, sin for given positions. Only rope_dim-sized."""
        if position_ids.dim() == 2:
            pos_ids = position_ids.squeeze(0)
        else:
            pos_ids = position_ids
        cos = self.cos_cached[:, pos_ids].to(MODEL_DTYPE)
        sin = self.sin_cached[:, pos_ids].to(MODEL_DTYPE)
        return cos, sin

    def forward_single(self, current_pos):
        """Get cos, sin for a single position."""
        cos = self.cos_cached[:, current_pos].view(1, 1, 1, self.rope_dim).to(MODEL_DTYPE)
        sin = self.sin_cached[:, current_pos].view(1, 1, 1, self.rope_dim).to(MODEL_DTYPE)
        return cos, sin


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(q, k, cos, sin, rope_dim):
    """Apply RoPE to only the first rope_dim dimensions of Q and K.

    q, k: [B, num_heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, rope_dim]
    """
    q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    x = hidden_states.unsqueeze(1)
    x = x.repeat(1, n_rep, 1, 1)
    return x.view(1, -1, x.size(-2), x.size(-1))


class Qwen35MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(config.intermediate_size, config.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)

    def forward(self, x):
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        a = self.gate_proj(x)
        b = self.up_proj(x)
        c = F.silu(a)
        d = c * b
        e = self.down_proj(d)
        return e.squeeze(2).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Gated DeltaNet (Linear Attention)
# ---------------------------------------------------------------------------

class Qwen35GatedDeltaNet(nn.Module):
    """Gated DeltaNet layer — recurrent linear attention with fixed-size state."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.linear_num_key_heads
        self.d_k = config.linear_key_head_dim
        self.d_v = config.linear_value_head_dim
        self.q_dim = self.num_heads * self.d_k
        self.k_dim = self.num_heads * self.d_k
        self.v_dim = self.num_heads * self.d_v
        self.qkv_dim = self.q_dim + self.k_dim + self.v_dim
        self.conv_kernel = config.linear_conv_kernel_dim

        # Projections (all Conv2d for ANE)
        self.in_proj_qkv = nn.Conv2d(config.hidden_size, self.qkv_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_z = nn.Conv2d(config.hidden_size, self.v_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_a = nn.Conv2d(config.hidden_size, self.num_heads, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_b = nn.Conv2d(config.hidden_size, self.num_heads, 1, bias=False, dtype=MODEL_DTYPE)
        self.out_proj = nn.Conv2d(self.v_dim, config.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)

        # CausalConv1D weight stored as parameter (manual computation avoids cat on ANE)
        # Shape [1, qkv_dim, 1, kernel] to broadcast correctly with [1, qkv_dim, 1, S] inputs
        self.conv1d_weight = nn.Parameter(
            torch.randn(1, self.qkv_dim, 1, self.conv_kernel, dtype=MODEL_DTYPE) * 0.01
        )

        # DeltaNet parameters
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=MODEL_DTYPE))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads, dtype=MODEL_DTYPE))

        # GatedRMSNorm (per-head, weight size = d_v, not v_dim)
        self.gated_rmsnorm = Qwen35GatedRMSNorm(self.d_v, eps=config.rms_norm_eps)

    def _deltanet_step(self, state, q, k, v, beta, gate):
        """Single DeltaNet recurrence step. Runs in float32 for numerical stability.

        Uses TRANSPOSED storage layout [H, d_v, d_k] to avoid state.transpose()
        which breaks ANE MLState. All transposes are on regular tensors (k, q).

        Args:
            state: [1, H, d_v, d_k]  (transposed storage)
            q: [1, H, d_k, 1]
            k: [1, H, d_k, 1]
            v: [1, H, d_v, 1]
            beta: [1, H, 1, 1]
            gate: [1, H, 1, 1]
        Returns:
            output: [1, H, d_v, 1]
            new_state: [1, H, d_v, d_k]  (transposed storage)
        """
        orig_dtype = q.dtype
        state = state.float()
        q, k, v, beta, gate = q.float(), k.float(), v.float(), beta.float(), gate.float()

        # Decay old memory
        state = torch.exp(gate) * state

        # Query stored value: state @ k (no transpose needed with transposed storage)
        # [1, H, d_v, d_k] @ [1, H, d_k, 1] = [1, H, d_v, 1]
        retrieved = torch.matmul(state, k)

        # Error correction
        delta = beta * (v - retrieved)  # [1, H, d_v, 1]

        # Write correction: state += delta @ k^T (transpose on k, not state)
        # [1, H, d_v, 1] @ [1, H, 1, d_k] = [1, H, d_v, d_k]
        state = state + torch.matmul(delta, k.transpose(-2, -1))

        # Read output: state @ q (no transpose needed)
        # [1, H, d_v, d_k] @ [1, H, d_k, 1] = [1, H, d_v, 1]
        output = torch.matmul(state, q)

        return output.to(orig_dtype), state.to(orig_dtype)

    def _conv1d_decode(self, qkv, conv_state_layer):
        """Manual CausalConv1D for single-token decode — avoids cat on ANE.

        qkv: [1, qkv_dim, 1, 1]
        conv_state_layer: [1, qkv_dim, 1, CONV_STATE_PAD]  (padded, only first 3 slots used)
        Returns: conv_out [1, qkv_dim, 1, 1], new_conv_state [1, qkv_dim, 1, CONV_STATE_PAD]
        """
        # Manual convolution: sum of weight_i * input_i (only first 3 slots of padded buffer)
        conv_out = (
            self.conv1d_weight[:, :, :, 0:1] * conv_state_layer[:, :, :, 0:1] +
            self.conv1d_weight[:, :, :, 1:2] * conv_state_layer[:, :, :, 1:2] +
            self.conv1d_weight[:, :, :, 2:3] * conv_state_layer[:, :, :, 2:3] +
            self.conv1d_weight[:, :, :, 3:4] * qkv
        )

        # Update conv_state: shift left within first 3 slots, write new value
        new_conv_state = conv_state_layer.clone()
        new_conv_state[:, :, :, 0:2] = conv_state_layer[:, :, :, 1:3]
        new_conv_state[:, :, :, 2:3] = qkv

        return conv_out, new_conv_state

    def forward_regular(self, hidden_states, delta_state_layer, conv_state_layer):
        """Forward pass for single-token decode.

        hidden_states: [1, 1, hidden_size]
        delta_state_layer: [1, num_heads, d_k, d_v]
        conv_state_layer: [1, qkv_dim, 1, kernel-1]
        Returns: output [1, 1, hidden_size], new_delta_state, new_conv_state
        """
        # Project to 4D for Conv2d
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)  # [1, hidden, 1, 1]

        qkv = self.in_proj_qkv(hs)  # [1, qkv_dim, 1, 1]
        z = self.in_proj_z(hs)       # [1, v_dim, 1, 1]
        a = self.in_proj_a(hs)       # [1, num_heads, 1, 1]
        b = self.in_proj_b(hs)       # [1, num_heads, 1, 1]

        # CausalConv1D (manual for decode)
        qkv, new_conv_state = self._conv1d_decode(qkv, conv_state_layer)

        # SiLU activation
        qkv = F.silu(qkv)

        # Split QKV
        q = qkv[:, :self.q_dim, :, :]
        k = qkv[:, self.q_dim:self.q_dim + self.k_dim, :, :]
        v = qkv[:, self.q_dim + self.k_dim:, :, :]

        # Reshape to per-head: [1, H, d_k/d_v, 1]
        q = q.view(1, self.num_heads, self.d_k, 1)
        k = k.view(1, self.num_heads, self.d_k, 1)
        v = v.view(1, self.num_heads, self.d_v, 1)

        # L2 normalize Q and K (eps=1e-6 to match HF's l2norm)
        q = q * torch.rsqrt(torch.sum(q * q, dim=2, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(torch.sum(k * k, dim=2, keepdim=True) + 1e-6)

        # Scale query by 1/sqrt(d_k) — matches HF's recurrence scaling
        q = q * (1.0 / math.sqrt(self.d_k))

        # Compute gate and beta (gate computed in float32 for numerical stability, per HF reference)
        beta = torch.sigmoid(b)  # [1, num_heads, 1, 1]
        # Manual softplus: log(1+exp(x)) — avoids F.softplus's threshold optimization
        # which generates greater_equal + select ops that block ANE.
        # clamp also generates conditional ops, so we use min() with a constant instead.
        sp_input = a.float() + self.dt_bias.float().view(1, -1, 1, 1)
        sp_output = torch.log(1.0 + torch.exp(sp_input))
        gate = (-self.A_log.float().exp().view(1, -1, 1, 1) * sp_output).to(MODEL_DTYPE)

        # DeltaNet recurrence
        output, new_delta_state = self._deltanet_step(delta_state_layer, q, k, v, beta, gate)

        # GatedRMSNorm (per-head)
        # output: [1, H, d_v, 1], z: [1, v_dim, 1, 1]
        # Reshape for per-head norm: [B, H, 1, d_v] format (last dim = d_v for LayerNorm)
        output = output.squeeze(-1)  # [1, H, d_v]
        z_heads = z.view(1, self.num_heads, self.d_v, 1).squeeze(-1)  # [1, H, d_v]
        output = self.gated_rmsnorm(output, z_heads)  # [1, H, d_v]

        # Reshape to [1, v_dim, 1, 1] for output projection
        output = output.view(1, self.v_dim).unsqueeze(-1).unsqueeze(-1)  # [1, v_dim, 1, 1]
        result = self.out_proj(output)  # [1, hidden, 1, 1]
        result = result.squeeze(2).permute(0, 2, 1)  # [1, 1, hidden]

        return result, new_delta_state, new_conv_state

    def forward_prefill(self, hidden_states, delta_state_layer, conv_state_layer):
        """Forward pass for prefill — processes tokens sequentially for DeltaNet recurrence.

        hidden_states: [1, seq_len, hidden_size]
        Returns: output [1, seq_len, hidden_size], new_delta_state, new_conv_state
        """
        seq_len = hidden_states.shape[1]
        outputs = []

        state = delta_state_layer
        conv_st = conv_state_layer

        for t in range(seq_len):
            token = hidden_states[:, t:t+1, :]  # [1, 1, hidden]
            out, state, conv_st = self.forward_regular(token, state, conv_st)
            outputs.append(out)

        output = torch.cat(outputs, dim=1)  # [1, seq_len, hidden]
        return output, state, conv_st


# ---------------------------------------------------------------------------
# Full Attention (with sigmoid gate + partial RoPE)
# ---------------------------------------------------------------------------

class Qwen35Attention(nn.Module):
    """Full attention with sigmoid output gate and partial RoPE."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_dim = int(config.head_dim * config.partial_rotary_factor)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q projection is 2x when attn_output_gate=True (half for query, half for gate)
        q_proj_dim = self.num_heads * self.head_dim * (2 if config.attn_output_gate else 1)
        kv_proj_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Conv2d(config.hidden_size, q_proj_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.k_proj = nn.Conv2d(config.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.v_proj = nn.Conv2d(config.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.o_proj = nn.Conv2d(self.num_heads * self.head_dim, config.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)

        # Per-head Q/K normalization
        self.q_norm = Qwen35HeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35HeadNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_output_gate = config.attn_output_gate

    def _project_qkv(self, hidden_states):
        """Project hidden states to Q, K, V. Returns shapes ready for attention.

        hidden_states: [1, seq_len, hidden_size]
        """
        seq_len = hidden_states.shape[1]
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)  # [1, hidden, 1, seq_len]

        q_full = self.q_proj(hs)  # [1, q_proj_dim, 1, seq_len]
        key_states = self.k_proj(hs)  # [1, kv_dim, 1, seq_len]
        value_states = self.v_proj(hs)  # [1, kv_dim, 1, seq_len]

        # Split Q into query + gate if attn_output_gate.
        # HF interleaves per-head: [h0_q(head_dim), h0_gate(head_dim), h1_q, h1_gate, ...]
        # so we must reshape to per-head FIRST, then split along the inner dimension.
        if self.attn_output_gate:
            q_5d = q_full.view(1, self.num_heads, 2 * self.head_dim, 1, seq_len)
            query_states = q_5d[:, :, :self.head_dim, :, :].squeeze(3)   # [1, H, head_dim, S]
            gate = q_5d[:, :, self.head_dim:, :, :].squeeze(3)           # [1, H, head_dim, S]
            query_states = query_states.permute(0, 1, 3, 2)  # [1, H, S, head_dim]
            gate = gate.permute(0, 1, 3, 2)                  # [1, H, S, head_dim]
        else:
            query_states = q_full.view(1, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
            gate = None

        # Reshape K, V to [1, num_heads, seq_len, head_dim]
        key_states = key_states.view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        value_states = value_states.view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)

        # Apply Q/K normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        return query_states, key_states, value_states, gate

    def forward_regular(self, hidden_states, kv_cache_layer, causal_mask, current_pos, rotary_emb):
        """Single-token decode with KV cache.

        hidden_states: [1, 1, hidden_size]
        kv_cache_layer: (K_cache, V_cache) each [num_kv_heads, state_length, head_dim]
        """
        query_states, key_states, value_states, gate = self._project_qkv(hidden_states)

        # Apply partial RoPE
        cos, sin = rotary_emb  # [1, 1, 1, rope_dim]
        query_states, key_states = apply_partial_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_dim
        )

        # Get KV cache
        K_cache, V_cache = kv_cache_layer
        K_cache = K_cache[..., :self.config.state_length, :]
        V_cache = V_cache[..., :self.config.state_length, :]

        # Repeat KV for GQA
        n_rep = self.num_heads // self.num_kv_heads
        key_full = repeat_kv(K_cache, n_rep)
        value_full = repeat_kv(V_cache, n_rep)

        # Attention
        attn_weights = torch.matmul(query_states.to(MODEL_DTYPE), key_full.transpose(-1, -2).to(MODEL_DTYPE)) * self.scale
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)[:, :, :1, :self.config.state_length]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_full.to(MODEL_DTYPE))

        # Sigmoid gate
        if self.attn_output_gate and gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(1, 1, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)

    def forward_prefill(self, hidden_states, kv_cache_layer, causal_mask, rotary_emb):
        """Prefill with full KV cache."""
        seq_len = hidden_states.shape[1]
        query_states, key_states, value_states, gate = self._project_qkv(hidden_states)

        # Apply partial RoPE
        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)  # [1, 1, seq_len, rope_dim]
        sin = sin.permute(0, 2, 1, 3)
        query_states, key_states = apply_partial_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_dim
        )

        # Get KV cache
        K_cache, V_cache = kv_cache_layer
        K_cache = K_cache[..., :self.config.state_length, :]
        V_cache = V_cache[..., :self.config.state_length, :]

        n_rep = self.num_heads // self.num_kv_heads
        key_full = repeat_kv(K_cache, n_rep)
        value_full = repeat_kv(V_cache, n_rep)

        attn_weights = torch.matmul(query_states.to(MODEL_DTYPE), key_full.transpose(-1, -2).to(MODEL_DTYPE)) * self.scale
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)[:, :, :seq_len, :self.config.state_length]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_full.to(MODEL_DTYPE))

        if self.attn_output_gate and gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        attn_output = attn_output.transpose(1, 2).contiguous().view(1, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.self_attn = Qwen35GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen35Attention(config, layer_idx)

        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


# ---------------------------------------------------------------------------
# Model (backbone with 3 state types)
# ---------------------------------------------------------------------------

class Qwen35Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=MODEL_DTYPE)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE (only for full attention layers)
        self.rotary_emb = Qwen35RotaryEmbedding(config)

        # ---- Per-layer state buffers for MLState compatibility ----
        # Each layer gets its own buffer (1 update per buffer per forward).
        # This avoids ANE error -14 from multiple slice updates on shared buffers.

        # Delta state: transposed storage [d_v, d_k] to avoid state.transpose()
        for i in range(config.num_linear_layers):
            self.register_buffer(f"delta_{i}", torch.zeros(
                1, config.linear_num_key_heads,
                config.linear_value_head_dim, config.linear_key_head_dim,
                dtype=MODEL_DTYPE, device=TEST_DEVICE,
            ))

        # Conv state: padded to CONV_STATE_PAD for ANE 32-byte alignment
        conv_kernel_minus1 = config.linear_conv_kernel_dim - 1  # actual = 3
        for i in range(config.num_linear_layers):
            self.register_buffer(f"conv_{i}", torch.zeros(
                1, config.linear_qkv_dim,
                1, CONV_STATE_PAD,
                dtype=MODEL_DTYPE, device=TEST_DEVICE,
            ))

        # KV cache: separate K and V buffers per attention layer
        for i in range(config.num_full_layers):
            self.register_buffer(f"kv_key_{i}", torch.zeros(
                1, config.num_key_value_heads,
                config.state_length, config.head_dim,
                dtype=MODEL_DTYPE, device=TEST_DEVICE,
            ))
            self.register_buffer(f"kv_val_{i}", torch.zeros(
                1, config.num_key_value_heads,
                config.state_length, config.head_dim,
                dtype=MODEL_DTYPE, device=TEST_DEVICE,
            ))

        # Store conv actual width for slicing during decode
        self.conv_actual_width = conv_kernel_minus1  # 3

        n_bufs = config.num_linear_layers * 2 + config.num_full_layers * 2
        total_mb = sum(b.numel() * 2 for _, b in self.named_buffers()
                       if not _.startswith("rotary")) / 1024 / 1024
        print(f"Qwen35Model initialized:")
        print(f"  {config.num_linear_layers} linear layers, {config.num_full_layers} full layers")
        print(f"  {n_bufs} state buffers, {total_mb:.1f} MB total")
        print(f"  delta: (1, {config.linear_num_key_heads}, {config.linear_value_head_dim}, {config.linear_key_head_dim}) × {config.num_linear_layers} [transposed]")
        print(f"  conv:  (1, {config.linear_qkv_dim}, 1, {CONV_STATE_PAD}) × {config.num_linear_layers} [padded]")
        print(f"  kv:    (1, {config.num_key_value_heads}, {config.state_length}, {config.head_dim}) × {config.num_full_layers*2} [K+V separate]")

    def get_rotary_embeddings_s(self, current_pos):
        """Get partial RoPE for single position."""
        return self.rotary_emb.forward_single(current_pos)

    def get_rotary_embedding_prefill(self, positions):
        """Get partial RoPE for a sequence of positions."""
        cos, sin = self.rotary_emb(positions)
        seq_len = cos.shape[1]
        rope_dim = self.rotary_emb.rope_dim
        cos = cos.view(1, seq_len, 1, rope_dim)
        sin = sin.view(1, seq_len, 1, rope_dim)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def process_layer(self, layer_idx, hidden_states, causal_mask, current_pos, rotary_emb,
                       IN_PREFILL=False, update_mask=None):
        """Process a single layer, routing to appropriate attention type.

        Args:
            update_mask: [1, 1, state_length, 1] with 1.0 at current_pos.
                When provided, uses mask-based KV cache writes instead of
                dynamic slicing (required for ANE compatibility).
        """
        layer = self.layers[layer_idx]
        state_type, state_idx = get_layer_state_mapping(layer_idx, self.config.layer_types)

        # Pre-attention norm
        normalized = layer.input_layernorm(hidden_states)

        if state_type == 'linear':
            # DeltaNet layer — per-layer buffers
            delta_buf = getattr(self, f"delta_{state_idx}")   # [1, H, d_v, d_k] transposed
            conv_buf = getattr(self, f"conv_{state_idx}")     # [1, qkv_dim, 1, CONV_STATE_PAD]

            if IN_PREFILL:
                attn_output, new_delta, new_conv = layer.self_attn.forward_prefill(
                    normalized, delta_buf, conv_buf
                )
            else:
                attn_output, new_delta, new_conv = layer.self_attn.forward_regular(
                    normalized, delta_buf, conv_buf
                )

            # Write back to per-layer buffers (single update each)
            delta_buf[0:1] = new_delta
            conv_buf[0:1] = new_conv

        else:
            # Full attention layer — per-layer K/V buffers
            kv_key_buf = getattr(self, f"kv_key_{state_idx}")  # [1, num_kv_heads, state_len, head_dim]
            kv_val_buf = getattr(self, f"kv_val_{state_idx}")

            K_cache = kv_key_buf.squeeze(0)
            V_cache = kv_val_buf.squeeze(0)

            if IN_PREFILL:
                # Get QKV for cache update
                query_states, key_states, value_states, _ = layer.self_attn._project_qkv(normalized)
                cos, sin = rotary_emb
                cos_p = cos.permute(0, 2, 1, 3)
                sin_p = sin.permute(0, 2, 1, 3)
                query_states, key_states = apply_partial_rotary_pos_emb(
                    query_states, key_states, cos_p, sin_p, layer.self_attn.rope_dim
                )

                seq_len = key_states.shape[2]
                # Write to KV cache
                kv_key_buf[:, :, current_pos:current_pos+seq_len, :] = key_states
                kv_val_buf[:, :, current_pos:current_pos+seq_len, :] = value_states

                # Re-read cache for attention
                K_cache = kv_key_buf.squeeze(0)
                V_cache = kv_val_buf.squeeze(0)

                attn_output = layer.self_attn.forward_prefill(
                    normalized, (K_cache, V_cache), causal_mask, rotary_emb
                )
            else:
                # Single token: project and write to cache
                query_states, key_states, value_states, _ = layer.self_attn._project_qkv(normalized)
                cos, sin = rotary_emb
                query_states, key_states = apply_partial_rotary_pos_emb(
                    query_states, key_states, cos, sin, layer.self_attn.rope_dim
                )

                if update_mask is not None:
                    # Mask-based write (ANE compatible — no dynamic slicing)
                    kv_key_buf[0:1] = kv_key_buf * (1.0 - update_mask) + key_states * update_mask
                    kv_val_buf[0:1] = kv_val_buf * (1.0 - update_mask) + value_states * update_mask
                else:
                    # Dynamic slicing (PyTorch mode)
                    pos = current_pos
                    kv_key_buf[:, :, pos:pos+1, :] = key_states
                    kv_val_buf[:, :, pos:pos+1, :] = value_states

                K_cache = kv_key_buf.squeeze(0)
                V_cache = kv_val_buf.squeeze(0)

                attn_output = layer.self_attn.forward_regular(
                    normalized, (K_cache, V_cache), causal_mask, current_pos, rotary_emb
                )

        # Residual
        hidden_states = hidden_states + attn_output

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def process_layers(self, hidden_states, causal_mask, current_pos, rotary_emb,
                        start_layer=0, end_layer=None, IN_PREFILL=False, update_mask=None):
        """Process a range of transformer layers (for chunked conversion)."""
        if end_layer is None:
            end_layer = len(self.layers)

        for i in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                i, hidden_states, causal_mask, current_pos, rotary_emb,
                IN_PREFILL, update_mask=update_mask,
            )
        return hidden_states

    def forward(self, input_ids, causal_mask, position_ids, current_pos, IN_PREFILL=False, update_mask=None):
        hidden_states = self.embed_tokens(input_ids)

        if IN_PREFILL:
            rotary_emb = self.get_rotary_embedding_prefill(position_ids)
        else:
            rotary_emb = self.get_rotary_embeddings_s(current_pos)

        hidden_states = self.process_layers(
            hidden_states, causal_mask, current_pos, rotary_emb,
            start_layer=0, end_layer=None, IN_PREFILL=IN_PREFILL,
            update_mask=update_mask,
        )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_pretrained_weights(self, model_path):
        """Load weights from HuggingFace Qwen3.5 checkpoint."""
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)

        state_dict = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))

        conv_state = {}
        for k, v in state_dict.items():
            # Strip HF prefix
            new_k = k
            if new_k.startswith("model.language_model."):
                new_k = new_k[len("model.language_model."):]
            elif new_k.startswith("model."):
                new_k = new_k[len("model."):]

            # Skip non-text weights
            if any(skip in k for skip in ["visual.", "mtp.", "lm_head."]):
                continue

            # Map DeltaNet attention keys
            new_k = new_k.replace("linear_attn.", "self_attn.")

            # Handle conv1d weight: (channels, 1, kernel) → parameter (1, channels, 1, kernel)
            if "conv1d.weight" in new_k:
                new_k = new_k.replace("conv1d.weight", "conv1d_weight")
                # HF format: (C, 1, K) → our format: (1, C, 1, K)
                conv_state[new_k] = v.unsqueeze(0).to(MODEL_DTYPE)
                continue

            # Handle A_log and dt_bias (1D parameters)
            if "A_log" in new_k or "dt_bias" in new_k:
                conv_state[new_k] = v.to(MODEL_DTYPE)
                continue

            # Handle GatedRMSNorm weight
            if ".norm.weight" in new_k and "layernorm" not in new_k:
                new_k = new_k.replace(".norm.weight", ".gated_rmsnorm.weight")
                conv_state[new_k] = v.to(MODEL_DTYPE)
                continue

            # Reshape projection weights for Conv2d: (out, in) → (out, in, 1, 1)
            if any(proj in new_k for proj in [
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                "gate_proj.weight", "up_proj.weight", "down_proj.weight",
                "in_proj_qkv.weight", "in_proj_z.weight",
                "in_proj_a.weight", "in_proj_b.weight", "out_proj.weight",
            ]):
                conv_state[new_k] = v.to(MODEL_DTYPE).view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[new_k] = v.to(MODEL_DTYPE)

        missing, unexpected = self.load_state_dict(conv_state, strict=False)

        # Filter expected missing keys (per-layer state buffers + rotary)
        expected_missing_prefixes = ("delta_", "conv_", "kv_key_", "kv_val_", "rotary_emb.inv_freq")
        missing = [m for m in missing if not any(m.startswith(p) or p in m for p in expected_missing_prefixes)]

        allow_missing = os.environ.get("ANEMLL_ALLOW_MISSING_WEIGHTS", "").lower() in ("1", "true", "yes")
        if missing:
            print(f"Missing keys ({len(missing)}):", missing[:10])
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            if not allow_missing:
                return False
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}):", unexpected[:10])
        return True


# ---------------------------------------------------------------------------
# Top-level CausalLM
# ---------------------------------------------------------------------------

class Qwen35ForCausalLM(nn.Module):
    config_class = Qwen35Config

    def __init__(self, config, enable_coreml=False, **kwargs):
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml

        if enable_coreml:
            global ENABLE_COREML
            ENABLE_COREML = True

        self.model = Qwen35Model(config)

        # 16-way split LM head (248320 vocab)
        if ENABLE_CONV2D and ENABLE_VACAB_SPLIT16:
            vocab_split = config.vocab_size // 16
            vocab_remainder = config.vocab_size % 16
            for i in range(16):
                split_size = vocab_split + (1 if i < vocab_remainder else 0)
                setattr(self, f"lm_head16_{i+1}",
                        nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE))
            print(f"Created 16-way split LM head (vocab={config.vocab_size}, splits of ~{vocab_split})")
        else:
            self.lm_head = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE)

    def forward(self, input_ids, update_mask, position_ids, causal_mask, current_pos, IN_PREFILL=False):
        hidden_states = self.model(input_ids, causal_mask, position_ids, current_pos, IN_PREFILL=IN_PREFILL)

        # Extract last position for LM head
        if not IN_PREFILL and current_pos is not None:
            seq_len = hidden_states.shape[1]
            if seq_len == 1:
                pos_tensor = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
            else:
                pos_tensor = current_pos if isinstance(current_pos, torch.Tensor) else torch.tensor([current_pos])
            hidden_states = torch.index_select(hidden_states, dim=1, index=pos_tensor)

        # Project to vocab
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        if ENABLE_CONV2D and ENABLE_VACAB_SPLIT16:
            logits_parts = []
            for i in range(16):
                part = getattr(self, f"lm_head16_{i+1}")(hidden_states).squeeze(2).transpose(1, 2)
                logits_parts.append(part)

            if self.enable_coreml and ENABLE_LOGITS2:
                return tuple(logits_parts)
            else:
                logits = torch.cat(logits_parts, dim=2)
        else:
            logits = self.lm_head(hidden_states).squeeze(2).transpose(1, 2)

        return logits

    def prefill_kv_cache(self, input_ids, position_ids, start_pos, causal_mask):
        """Pre-fill caches for a batch of tokens."""
        hidden_states = self.model.embed_tokens(input_ids).to(MODEL_DTYPE)
        if causal_mask is not None:
            seq_length = input_ids.shape[1]
            causal_mask = causal_mask[:, :, :seq_length, :]

        rotary_emb = self.model.get_rotary_embedding_prefill(position_ids)
        for i in range(len(self.model.layers)):
            hidden_states = self.model.process_layer(
                i, hidden_states, causal_mask, start_pos, rotary_emb, IN_PREFILL=True
            )

    def load_pretrained_weights(self, model_path):
        """Load full model weights including LM head."""
        if not self.model.load_pretrained_weights(model_path):
            return False

        # Load LM head from embed_tokens (tie_word_embeddings=True)
        state_dict = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))

        # Find LM head weight
        lm_head_weight = None
        for k, v in state_dict.items():
            if k == "lm_head.weight":
                lm_head_weight = v
                break

        if lm_head_weight is None and self.config.tie_word_embeddings:
            # Use embed_tokens weight
            for k, v in state_dict.items():
                if "embed_tokens.weight" in k:
                    lm_head_weight = v
                    print("Using embed_tokens.weight for LM head (tie_word_embeddings=True)")
                    break

        if lm_head_weight is None:
            print("ERROR: Cannot find LM head weight")
            return False

        lm_head_weight = lm_head_weight.to(MODEL_DTYPE)
        reshaped = lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1)

        if ENABLE_CONV2D and ENABLE_VACAB_SPLIT16:
            vocab_split = self.config.vocab_size // 16
            vocab_remainder = self.config.vocab_size % 16
            split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
            splits = torch.split(reshaped, split_sizes)
            for i, split in enumerate(splits):
                getattr(self, f"lm_head16_{i+1}").weight.data.copy_(split)
        else:
            self.lm_head.weight.data.copy_(reshaped)

        return True

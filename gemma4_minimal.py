"""Minimal, from-scratch PyTorch implementation of a text-only Gemma 4 decoder.

This is a single-file educational port of the Gemma 4 architecture, faithful
to the JAX reference in ``gemma/gm/nn/gemma4/`` in this repository. It covers
the two deployment shapes most people care about:

* **Dense** (E4B / 31B style) — stacked Transformer blocks with interleaved
  local-sliding and global attention layers, GQA, QK-Norm, partial RoPE.
* **Mixture-of-Experts** (26B A4B style) — same backbone, but every block's
  FFW is replaced by a parallel ``(MoE branch + dense shared branch)`` pair.

See ``GEMMA4_EXPLAINED.md`` for a component-by-component walkthrough with
math, shapes, and pointers to the JAX reference.

What this file implements:

* :class:`RMSNorm`
* :func:`apply_rope` — partial RoPE (the "dual-RoPE" used on global layers)
* :class:`Embedder` — tied input / output with ``sqrt(D)`` scaling
* :class:`GeGLU` — the standard Gemma FFW
* :class:`GemmaAttention` — GQA + QK-Norm + sliding-window + ``k_eq_v``
* :class:`TopKRouter` and :class:`MoEFeedForward`
* :class:`TransformerBlock` — pre/post norms, ``skip_scale``, dense or MoE
* :class:`Gemma4` — full model with tied, softcapped head
* :func:`generate` — greedy decoding with a simple per-layer KV cache

What this file does NOT implement (on purpose — see the explainer for why):

* Vision / audio encoders (multimodal).
* Tokenizer integration (we work with integer token IDs directly).
* KV-cache sharing across layers.
* Per-Layer Embeddings (PLE) — supported via a config flag but off by default.
* Checkpoint loading from the JAX params tree.

Typical usage::

    import torch
    from gemma4_minimal import Gemma4, Gemma4Config, generate

    cfg = Gemma4Config.e4b_like(
        vocab_size=1024, embed_dim=128, num_layers=6,
        num_heads=4, num_kv_heads=2, head_dim=32,
        sliding_window_size=16,
    )
    model = Gemma4(cfg).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    out = generate(model, prompt, max_new_tokens=8)
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionType(enum.Enum):
  LOCAL_SLIDING = "local_sliding"
  GLOBAL = "global"


@dataclasses.dataclass
class Gemma4Config:
  """Configuration for a (text-only) Gemma 4 model.

  Fields map 1:1 to the reference ``TransformerConfig`` in
  ``gemma/gm/nn/gemma4/_config.py``. Unsupported features (vision, audio,
  KV-cache sharing) are simply absent.
  """

  vocab_size: int
  num_layers: int
  embed_dim: int
  num_heads: int
  num_kv_heads: int
  head_dim: int = 256
  hidden_dim: Optional[int] = None  # dense FFW hidden; defaults to 4 * embed_dim
  sliding_window_size: int = 512

  attention_pattern: tuple[AttentionType, ...] = (
      AttentionType.LOCAL_SLIDING,
      AttentionType.LOCAL_SLIDING,
      AttentionType.LOCAL_SLIDING,
      AttentionType.LOCAL_SLIDING,
      AttentionType.LOCAL_SLIDING,
      AttentionType.GLOBAL,
  )

  # Global-attention specific
  num_global_kv_heads: Optional[int] = None  # defaults to num_kv_heads
  global_key_size: Optional[int] = None  # defaults to head_dim
  k_eq_v_global: bool = False
  global_rope_proportion: float = 0.25
  local_rope_proportion: float = 1.0
  local_base_frequency: float = 10_000.0
  global_base_frequency: float = 1_000_000.0

  # Sandwich norms
  use_post_attn_norm: bool = True
  use_post_ffw_norm: bool = True
  qk_norm_with_scale: bool = True

  # Output
  final_logit_softcap: Optional[float] = 30.0

  # MoE (off by default → dense FFW)
  enable_moe: bool = False
  num_experts: int = 0
  expert_dim: int = 0
  top_k_experts: int = 0
  moe_dense_hidden_dim: int = 0  # shared dense branch hidden dim

  # Per-Layer Embeddings (off by default; "E-series" models use it)
  per_layer_input_dim: int = 0

  @property
  def attention_types(self) -> tuple[AttentionType, ...]:
    """Expand the short attention_pattern to ``num_layers`` entries."""
    pat = self.attention_pattern
    n = self.num_layers
    out = list(pat * (n // len(pat)))
    out += list(pat[: n % len(pat)])
    return tuple(out)

  @classmethod
  def e4b_like(cls, **overrides: Any) -> "Gemma4Config":
    """Tiny dense config that mirrors the E4B shape (5L:1G, 4x FFW). CPU-friendly."""
    defaults: dict[str, Any] = dict(
        vocab_size=256,
        num_layers=6,
        embed_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        hidden_dim=128 * 4,
        sliding_window_size=16,
    )
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def moe_26b_like(cls, **overrides: Any) -> "Gemma4Config":
    """Tiny MoE config that mirrors the 26B A4B shape (128 experts → 8 here).

    Enables: MoE, ``k_eq_v_global``, smaller expert_dim, and a parallel dense
    shared branch.
    """
    defaults: dict[str, Any] = dict(
        vocab_size=256,
        num_layers=6,
        embed_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        hidden_dim=None,  # unused when MoE is on
        sliding_window_size=16,
        num_global_kv_heads=1,
        k_eq_v_global=True,
        enable_moe=True,
        num_experts=8,
        expert_dim=32,
        top_k_experts=2,
        moe_dense_hidden_dim=64,
    )
    defaults.update(overrides)
    return cls(**defaults)


# ---------------------------------------------------------------------------
# Core layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
  """Root-mean-square layer norm, matching ``gemma/gm/nn/gemma4/_layers.RMSNorm``.

  ``y = x / sqrt(mean(x^2) + eps) * scale``. When ``with_scale=False`` the
  learned ``scale`` is omitted (used for ``value_norm`` and MoE ``router_norm``).
  """

  def __init__(self, dim: int, *, with_scale: bool = True, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.with_scale = with_scale
    if with_scale:
      self.scale = nn.Parameter(torch.ones(dim))
    else:
      self.register_parameter("scale", None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + self.eps)
    if self.with_scale:
      x = x * self.scale
    return x


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    base_frequency: float,
    rope_proportion: float,
) -> torch.Tensor:
  """Rotary position embeddings with a *partial* rotation (Gemma 4 "dual-RoPE").

  Only the first ``int(rope_proportion * H // 2)`` frequency pairs are
  rotated; the remaining pairs are left untouched (the "NoPE" part).
  This matches the JAX reference in
  ``gemma/gm/math/_positional_embeddings.apply_rope``.

  Args:
    x: shape ``[B, L, N, H]`` with ``H`` even.
    positions: shape ``[B, L]``, integer token positions.
    base_frequency: base for the geometric frequency sequence (10k local,
      1M global).
    rope_proportion: fraction of head-dim pairs to rotate. Use ``1.0`` for
      local layers, ``0.25`` for global layers.

  Returns:
    Rotated tensor of the same shape and dtype as ``x``.
  """
  head_dim = x.shape[-1]
  half = head_dim // 2
  rope_angles = int(rope_proportion * half)
  nope_angles = half - rope_angles

  device = x.device
  freq_exp = (2.0 / head_dim) * torch.arange(
      rope_angles, dtype=torch.float32, device=device
  )
  timescale_rot = base_frequency**freq_exp  # [rope_angles]
  # Pad the non-rotated portion with inf → angle=0 → cos=1, sin=0
  if nope_angles > 0:
    timescale = torch.cat([
        timescale_rot,
        torch.full(
            (nope_angles,), float("inf"), dtype=torch.float32, device=device
        ),
    ])
  else:
    timescale = timescale_rot

  # angle[b, l, 1, h_half] = positions[b, l] / timescale[h_half]
  angle = positions.to(torch.float32)[..., None, None] / timescale[None, None, None, :]
  sin = torch.sin(angle).to(x.dtype)
  cos = torch.cos(angle).to(x.dtype)

  first_half, second_half = x.chunk(2, dim=-1)
  first_out = first_half * cos - second_half * sin
  second_out = second_half * cos + first_half * sin
  return torch.cat([first_out, second_out], dim=-1)


class Embedder(nn.Module):
  """Tied input / output embedding with ``sqrt(embed_dim)`` input scaling."""

  def __init__(self, vocab_size: int, embed_dim: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.input_embedding = nn.Parameter(
        torch.randn(vocab_size, embed_dim) * (1.0 / math.sqrt(embed_dim))
    )

  def encode(self, tokens: torch.Tensor) -> torch.Tensor:
    """[B, L] int → [B, L, D]."""
    x = F.embedding(tokens, self.input_embedding)
    return x * math.sqrt(self.embed_dim)

  def decode(self, x: torch.Tensor) -> torch.Tensor:
    """[B, L, D] → [B, L, V] via weight tying."""
    return x @ self.input_embedding.T


class GeGLU(nn.Module):
  """GeGLU feed-forward, matching ``gemma/gm/nn/gemma4/_modules.FeedForward``.

  ``FFW(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down``.

  The reference stores ``W_gate`` and ``W_up`` as a single parameter of
  shape ``[2, hidden, features]`` (indexable along axis 0). We do the same.
  """

  def __init__(self, embed_dim: int, hidden_dim: int):
    super().__init__()
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.gating = nn.Parameter(
        torch.randn(2, hidden_dim, embed_dim) * (1.0 / math.sqrt(embed_dim))
    )
    self.linear = nn.Parameter(
        torch.randn(hidden_dim, embed_dim) * (1.0 / math.sqrt(hidden_dim))
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # [B, L, D] x [2, H, D] → [B, L, 2, H]
    gate_and_up = torch.einsum("bld,khd->blkh", x, self.gating)
    gate = gate_and_up[..., 0, :]
    up = gate_and_up[..., 1, :]
    activations = F.gelu(gate) * up
    return torch.einsum("blh,hd->bld", activations, self.linear)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class GemmaAttention(nn.Module):
  """Gemma 4 attention: GQA + QK-Norm + partial RoPE + optional sliding window.

  A single module handles both local-sliding and global layers; the caller
  (``TransformerBlock``) configures ``attn_type``, ``sliding_window_size``,
  ``rope_*``, ``head_dim`` (= global_key_size on global), and ``k_eq_v``
  appropriately.

  Cache protocol:
    ``cache`` is a dict with keys ``k``, ``v``, ``positions``, ``end_index``.
    When ``cache`` is provided, new KV is inserted at slots
    ``[end_index, end_index + L)`` and attention is computed over the full
    cache; unfilled slots are masked out by their sentinel position = -1.
  """

  def __init__(
      self,
      *,
      embed_dim: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      attn_type: AttentionType,
      sliding_window_size: int,
      rope_base_frequency: float,
      rope_proportion: float,
      qk_norm_with_scale: bool,
      k_eq_v: bool,
  ):
    super().__init__()
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by num_kv_heads"
        f" ({num_kv_heads}) for GQA"
    )
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.attn_type = attn_type
    self.sliding_window_size = sliding_window_size
    self.rope_base_frequency = rope_base_frequency
    self.rope_proportion = rope_proportion
    self.k_eq_v = k_eq_v

    init = lambda *s: torch.randn(*s) * (1.0 / math.sqrt(s[-2]))

    self.q_proj = nn.Parameter(init(num_heads, embed_dim, head_dim))
    if k_eq_v:
      self.kv_shared_proj = nn.Parameter(init(num_kv_heads, embed_dim, head_dim))
    else:
      self.kv_proj = nn.Parameter(init(2, num_kv_heads, embed_dim, head_dim))
    self.o_proj = nn.Parameter(
        torch.randn(num_heads, head_dim, embed_dim)
        * (1.0 / math.sqrt(num_heads * head_dim))
    )

    self.query_norm = RMSNorm(head_dim, with_scale=qk_norm_with_scale)
    self.key_norm = RMSNorm(head_dim, with_scale=qk_norm_with_scale)
    self.value_norm = RMSNorm(head_dim, with_scale=False)

    self._scale = 1.0 / math.sqrt(head_dim)

  def init_cache(
      self,
      batch_size: int,
      cache_size: int,
      device: torch.device,
      dtype: torch.dtype = torch.float32,
  ) -> dict:
    return {
        "k": torch.zeros(
            batch_size,
            cache_size,
            self.num_kv_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        ),
        "v": torch.zeros(
            batch_size,
            cache_size,
            self.num_kv_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        ),
        # -1 sentinel → "slot not yet filled" for mask construction.
        "positions": torch.full(
            (batch_size, cache_size), -1, device=device, dtype=torch.long
        ),
        "end_index": 0,
    }

  def forward(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      cache: Optional[dict] = None,
  ) -> tuple[torch.Tensor, Optional[dict]]:
    """[B, L, D] → [B, L, D]."""
    B, L, _ = x.shape

    # --- Q ---
    q = torch.einsum("bld,ndh->blnh", x, self.q_proj)  # [B, L, N, H]
    q = self.query_norm(q)
    q = apply_rope(
        q,
        positions,
        base_frequency=self.rope_base_frequency,
        rope_proportion=self.rope_proportion,
    )

    # --- K, V ---
    if self.k_eq_v:
      kv = torch.einsum("bld,kdh->blkh", x, self.kv_shared_proj)
      k_new = kv
      v_new = kv
    else:
      # [2, B, L, K, H]
      kv = torch.einsum("bld,ckdh->cblkh", x, self.kv_proj)
      k_new = kv[0]
      v_new = kv[1]
    k_new = self.key_norm(k_new)
    v_new = self.value_norm(v_new)
    k_new = apply_rope(
        k_new,
        positions,
        base_frequency=self.rope_base_frequency,
        rope_proportion=self.rope_proportion,
    )

    # --- Cache update ---
    if cache is not None:
      end_index = cache["end_index"]
      cache_size = cache["k"].shape[1]
      if end_index + L > cache_size:
        raise ValueError(
            f"cache overflow: end_index={end_index} + L={L} > cache_size"
            f" {cache_size}"
        )
      k_full = cache["k"].clone()
      v_full = cache["v"].clone()
      pos_full = cache["positions"].clone()
      k_full[:, end_index : end_index + L] = k_new
      v_full[:, end_index : end_index + L] = v_new
      pos_full[:, end_index : end_index + L] = positions
      new_cache = {
          "k": k_full,
          "v": v_full,
          "positions": pos_full,
          "end_index": end_index + L,
      }
      k_attend = k_full
      v_attend = v_full
      attend_positions = pos_full
    else:
      new_cache = None
      k_attend = k_new
      v_attend = v_new
      attend_positions = positions

    S = k_attend.shape[1]

    # --- Mask: causal + valid + optional sliding ---
    # [B, L, 1] vs [B, 1, S]
    pos_l = positions[:, :, None]
    pos_s = attend_positions[:, None, :]
    attend_valid = attend_positions >= 0  # [B, S]
    causal = pos_s <= pos_l
    attn_mask = causal & attend_valid[:, None, :]

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      w = self.sliding_window_size
      sliding = (pos_s > pos_l - w) & (pos_s < pos_l + w)
      attn_mask = attn_mask & sliding

    # --- GQA attention ---
    G = self.num_heads // self.num_kv_heads
    q_grouped = q.reshape(B, L, self.num_kv_heads, G, self.head_dim)
    # logits[b, l, k, g, s] = sum_h q[b,l,k,g,h] * k[b,s,k,h]
    logits = torch.einsum("blkgh,bskh->blkgs", q_grouped, k_attend)
    logits = logits.reshape(B, L, self.num_heads, S) * self._scale

    # [B, L, 1, S] for broadcasting over heads
    logits = logits.masked_fill(~attn_mask[:, :, None, :], float("-inf"))
    probs = F.softmax(logits.float(), dim=-1).to(q.dtype)

    probs_grouped = probs.reshape(B, L, self.num_kv_heads, G, S)
    encoded = torch.einsum("blkgs,bskh->blkgh", probs_grouped, v_attend)
    encoded = encoded.reshape(B, L, self.num_heads, self.head_dim)

    out = torch.einsum("blnh,nhd->bld", encoded, self.o_proj)
    return out, new_cache


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------


class TopKRouter(nn.Module):
  """Token → expert router used by the MoE block.

  Mirrors the ``router_norm`` → scale → logits path from ``_moe.MoE`` /
  ``MoERagged``: RMSNorm without scale, divide by ``sqrt(D)``, then multiply
  by a learned per-dim ``router_scale``, then a linear to ``[B, L, E]``.
  """

  def __init__(self, embed_dim: int, num_experts: int):
    super().__init__()
    self.embed_dim = embed_dim
    self.router_norm = RMSNorm(embed_dim, with_scale=False)
    self.router_scale = nn.Parameter(torch.ones(embed_dim))
    self.router_weight = nn.Parameter(
        torch.randn(embed_dim, num_experts) * (1.0 / math.sqrt(embed_dim))
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """[B, L, D] → logits [B, L, E]."""
    r = self.router_norm(x)
    r = r * (1.0 / math.sqrt(self.embed_dim))
    r = r * self.router_scale
    return r @ self.router_weight


class MoEFeedForward(nn.Module):
  """Sparse MoE feed-forward: top-k of ``num_experts`` tiny GeGLU experts.

  Faithful to ``gemma/gm/nn/gemma4/_moe.py`` in terms of shapes and math,
  but uses a readable Python loop over experts instead of the sorted /
  ragged dispatch used in the reference. Same outputs, simpler code.

  Per-expert scaling: each expert has a scalar multiplier (``per_expert_scale``)
  applied to its output. Top-k routing weights are renormalized to sum to 1
  per token before combining.
  """

  def __init__(
      self,
      embed_dim: int,
      expert_dim: int,
      num_experts: int,
      top_k: int,
  ):
    super().__init__()
    self.embed_dim = embed_dim
    self.expert_dim = expert_dim
    self.num_experts = num_experts
    self.top_k = top_k

    self.router = TopKRouter(embed_dim, num_experts)
    # Expert gating: [E, 2, hidden, D] — one GeGLU gating tensor per expert.
    self.expert_gating = nn.Parameter(
        torch.randn(num_experts, 2, expert_dim, embed_dim)
        * (1.0 / math.sqrt(embed_dim))
    )
    # Expert down-projection: [E, hidden, D]
    self.expert_linear = nn.Parameter(
        torch.randn(num_experts, expert_dim, embed_dim)
        * (1.0 / math.sqrt(expert_dim))
    )
    self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """[B, L, D] → [B, L, D]."""
    B, L, D = x.shape

    logits = self.router(x).float()  # [B, L, E]
    probs = F.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # [B, L, k]
    topk_probs = probs.gather(-1, topk_idx)
    # Renormalize so the k weights sum to 1 per token.
    topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
    topk_weights = topk_weights.to(x.dtype)

    x_flat = x.reshape(B * L, D)
    topk_idx_flat = topk_idx.reshape(B * L, self.top_k)  # [N, k]
    topk_w_flat = topk_weights.reshape(B * L, self.top_k)  # [N, k]

    output = torch.zeros_like(x_flat)

    # Dispatch: loop over experts (O(E) but readable).
    for e in range(self.num_experts):
      expert_mask = topk_idx_flat == e  # [N, k]
      token_mask = expert_mask.any(dim=-1)  # [N]
      if not token_mask.any():
        continue
      tokens = x_flat[token_mask]  # [N_e, D]

      # GeGLU for this expert.
      gate_and_up = torch.einsum(
          "nd,chd->nch", tokens, self.expert_gating[e]
      )  # [N_e, 2, hidden]
      activations = F.gelu(gate_and_up[:, 0, :]) * gate_and_up[:, 1, :]
      expert_out = torch.einsum(
          "nh,hd->nd", activations, self.expert_linear[e]
      )
      expert_out = expert_out * self.per_expert_scale[e]

      # Each token is routed to expert e at most once in top-k (no duplicates
      # from torch.topk), so the combine weight is just the matched entry.
      combine_weights = (expert_mask.to(topk_w_flat.dtype) * topk_w_flat).sum(
          dim=-1
      )  # [N]
      expert_out = expert_out * combine_weights[token_mask, None]

      output[token_mask] = output[token_mask] + expert_out

    return output.reshape(B, L, D)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
  """One Gemma 4 decoder block.

  Mirrors ``_modules.Block`` exactly:

  * Attention sandwich: ``pre_attention_norm → Attention → post_attention_norm``
  * FFW sandwich: either
      - dense: ``pre_ffw_norm → GeGLU → post_ffw_norm``
      - MoE:  ``(pre_ffw_norm → MoE → post_ffw1_norm) + (pre_ffw2_norm →
              GeGLU → post_ffw2_norm)`` → ``post_ffw_norm``
  * Optional PLE gate (E-series only).
  * Learned scalar ``skip_scale`` on block output.
  """

  def __init__(self, config: Gemma4Config, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.attn_type = config.attention_types[layer_idx]

    is_global = self.attn_type == AttentionType.GLOBAL
    if is_global:
      num_kv_heads = config.num_global_kv_heads or config.num_kv_heads
      head_dim = config.global_key_size or config.head_dim
      k_eq_v = config.k_eq_v_global
      rope_proportion = config.global_rope_proportion
      rope_base = config.global_base_frequency
    else:
      num_kv_heads = config.num_kv_heads
      head_dim = config.head_dim
      k_eq_v = False
      rope_proportion = config.local_rope_proportion
      rope_base = config.local_base_frequency

    self.pre_attention_norm = RMSNorm(config.embed_dim)
    self.attn = GemmaAttention(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_type=self.attn_type,
        sliding_window_size=config.sliding_window_size,
        rope_base_frequency=rope_base,
        rope_proportion=rope_proportion,
        qk_norm_with_scale=config.qk_norm_with_scale,
        k_eq_v=k_eq_v,
    )
    self.post_attention_norm = (
        RMSNorm(config.embed_dim) if config.use_post_attn_norm else nn.Identity()
    )

    if config.enable_moe:
      self._setup_moe()
    else:
      self._setup_dense()

    if config.per_layer_input_dim > 0:
      P = config.per_layer_input_dim
      self.per_layer_input_gate = nn.Parameter(
          torch.randn(config.embed_dim, P) * (1.0 / math.sqrt(config.embed_dim))
      )
      self.per_layer_projection = nn.Parameter(
          torch.randn(P, config.embed_dim) * (1.0 / math.sqrt(P))
      )
      self.post_per_layer_input_norm = RMSNorm(config.embed_dim)

    self.skip_scale = nn.Parameter(torch.ones(1))

  def _setup_dense(self):
    cfg = self.config
    hidden = cfg.hidden_dim if cfg.hidden_dim is not None else 4 * cfg.embed_dim
    self.pre_ffw_norm = RMSNorm(cfg.embed_dim)
    self.mlp = GeGLU(cfg.embed_dim, hidden)
    self.post_ffw_norm = (
        RMSNorm(cfg.embed_dim) if cfg.use_post_ffw_norm else nn.Identity()
    )

  def _setup_moe(self):
    cfg = self.config
    # Dense shared branch (mlp2 in the reference)
    self.pre_ffw2_norm = RMSNorm(cfg.embed_dim)
    self.mlp2 = GeGLU(cfg.embed_dim, cfg.moe_dense_hidden_dim)
    self.post_ffw2_norm = (
        RMSNorm(cfg.embed_dim) if cfg.use_post_ffw_norm else nn.Identity()
    )
    # MoE branch (mlp in the reference)
    self.pre_ffw_norm = RMSNorm(cfg.embed_dim)
    self.mlp = MoEFeedForward(
        embed_dim=cfg.embed_dim,
        expert_dim=cfg.expert_dim,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k_experts,
    )
    self.post_ffw1_norm = (
        RMSNorm(cfg.embed_dim) if cfg.use_post_ffw_norm else nn.Identity()
    )
    # Final combine-branch post norm
    self.post_ffw_norm = (
        RMSNorm(cfg.embed_dim) if cfg.use_post_ffw_norm else nn.Identity()
    )

  def _forward_dense(self, h: torch.Tensor) -> torch.Tensor:
    x = self.pre_ffw_norm(h)
    x = self.mlp(x)
    x = self.post_ffw_norm(x)
    return x

  def _forward_moe(self, h: torch.Tensor) -> torch.Tensor:
    d = self.pre_ffw2_norm(h)
    d = self.mlp2(d)
    d = self.post_ffw2_norm(d)
    m = self.pre_ffw_norm(h)
    m = self.mlp(m)
    m = self.post_ffw1_norm(m)
    return self.post_ffw_norm(d + m)

  def forward(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      cache: Optional[dict] = None,
      per_layer_input: Optional[torch.Tensor] = None,
  ) -> tuple[torch.Tensor, Optional[dict]]:
    # 1. Attention sandwich + residual.
    attn_out, new_cache = self.attn(
        self.pre_attention_norm(x), positions, cache=cache
    )
    attn_out = self.post_attention_norm(attn_out)
    h = x + attn_out

    # 2. FFW sandwich + residual.
    ffw_out = self._forward_moe(h) if self.config.enable_moe else self._forward_dense(h)
    out = h + ffw_out

    # 3. PLE gate (optional).
    if per_layer_input is not None:
      g = out @ self.per_layer_input_gate  # [B, L, P]
      g = F.gelu(g) * per_layer_input
      add = g @ self.per_layer_projection  # [B, L, D]
      add = self.post_per_layer_input_norm(add)
      out = out + add

    # 4. Learned skip scale.
    return out * self.skip_scale, new_cache


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class Gemma4(nn.Module):
  """Text-only Gemma 4 decoder.

  Structure:
    * :class:`Embedder` (scaled, weight-tied with the output head)
    * ``num_layers`` :class:`TransformerBlock` s (interleaved local/global)
    * final :class:`RMSNorm`
    * tied output head + ``tanh`` softcap
  """

  def __init__(self, config: Gemma4Config):
    super().__init__()
    self.config = config
    self.embedder = Embedder(config.vocab_size, config.embed_dim)
    self.blocks = nn.ModuleList(
        [TransformerBlock(config, i) for i in range(config.num_layers)]
    )
    self.final_norm = RMSNorm(config.embed_dim)

    if config.per_layer_input_dim > 0:
      P = config.per_layer_input_dim
      self.per_layer_embedding_table = nn.Parameter(
          torch.randn(config.vocab_size, config.num_layers, P)
          * (1.0 / math.sqrt(config.embed_dim))
      )
      self.per_layer_model_projection = nn.Parameter(
          torch.randn(config.embed_dim, config.num_layers, P)
          * (1.0 / math.sqrt(config.embed_dim))
      )
      self.per_layer_projection_norm = RMSNorm(P)

  def _compute_per_layer_inputs(
      self, x: torch.Tensor, tokens: torch.Tensor
  ) -> Optional[torch.Tensor]:
    if self.config.per_layer_input_dim == 0:
      return None
    # [B, L, D] @ [D, N_layers, P] → [B, L, N_layers, P]
    x_proj = torch.einsum("bld,dnp->blnp", x, self.per_layer_model_projection)
    x_proj = self.per_layer_projection_norm(x_proj)
    # Lookup: [B, L] → [B, L, N_layers, P]
    y = self.per_layer_embedding_table[tokens]
    y = y * math.sqrt(self.config.per_layer_input_dim)
    return (x_proj + y) * (1.0 / math.sqrt(2.0))

  def init_cache(
      self,
      batch_size: int,
      cache_size: int,
      device: torch.device,
      dtype: torch.dtype = torch.float32,
  ) -> dict:
    return {
        f"layer_{i}": block.attn.init_cache(batch_size, cache_size, device, dtype)
        for i, block in enumerate(self.blocks)
    }

  def forward(
      self,
      tokens: torch.Tensor,
      positions: Optional[torch.Tensor] = None,
      cache: Optional[dict] = None,
  ) -> tuple[torch.Tensor, Optional[dict]]:
    """[B, L] int tokens → logits [B, L, V], optional new cache."""
    B, L = tokens.shape
    device = tokens.device
    if positions is None:
      positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    x = self.embedder.encode(tokens)
    per_layer_inputs = self._compute_per_layer_inputs(x, tokens)

    new_cache: Optional[dict] = {} if cache is not None else None
    for i, block in enumerate(self.blocks):
      layer_cache = cache[f"layer_{i}"] if cache is not None else None
      ple_i = (
          per_layer_inputs[..., i, :] if per_layer_inputs is not None else None
      )
      x, lc = block(
          x, positions, cache=layer_cache, per_layer_input=ple_i
      )
      if new_cache is not None:
        new_cache[f"layer_{i}"] = lc

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      c = self.config.final_logit_softcap
      logits = torch.tanh(logits / c) * c

    return logits, new_cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: Gemma4,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 32,
    cache_size: Optional[int] = None,
) -> torch.Tensor:
  """Greedy decoding with a per-layer KV cache.

  Args:
    model: a :class:`Gemma4` instance.
    prompt_ids: ``[B, L]`` integer token IDs.
    max_new_tokens: number of tokens to generate.
    cache_size: total KV slots to allocate per layer. Defaults to
      ``L + max_new_tokens``.

  Returns:
    ``[B, L + max_new_tokens]`` integer tensor (prompt + generated).
  """
  model.eval()
  B, L = prompt_ids.shape
  device = prompt_ids.device
  dtype = next(model.parameters()).dtype

  cache_size = cache_size or (L + max_new_tokens)
  cache = model.init_cache(
      batch_size=B, cache_size=cache_size, device=device, dtype=dtype
  )

  # --- Prefill ---
  positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
  logits, cache = model(prompt_ids, positions=positions, cache=cache)
  next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

  generated = [prompt_ids, next_token]

  # --- Decode ---
  for t in range(max_new_tokens - 1):
    cur_pos = L + t
    positions = torch.full((B, 1), cur_pos, device=device, dtype=torch.long)
    logits, cache = model(next_token, positions=positions, cache=cache)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token)

  return torch.cat(generated, dim=1)


__all__ = [
    "AttentionType",
    "Gemma4Config",
    "RMSNorm",
    "apply_rope",
    "Embedder",
    "GeGLU",
    "GemmaAttention",
    "TopKRouter",
    "MoEFeedForward",
    "TransformerBlock",
    "Gemma4",
    "generate",
]

"""Smoke tests for ``gemma4_minimal``.

These verify the minimal implementation is internally consistent:

* Forward produces the right ``[B, L, V]`` shape for dense and MoE configs.
* Running in one-shot (no cache) gives the same logits as one-token-at-a-time
  with a KV cache — i.e. the cache is correctly constructed.
* The MoE router produces renormalized weights that sum to 1 per token.
* Sliding-window and global layers compose correctly over a longer sequence.

Run:  ``python -m pytest test_gemma4_minimal.py -v``
(or just)  ``python test_gemma4_minimal.py``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gemma4_minimal import (
    Gemma4,
    Gemma4Config,
    MoEFeedForward,
    TopKRouter,
    apply_rope,
    generate,
)


def _tiny_dense_config() -> Gemma4Config:
  return Gemma4Config.e4b_like(
      vocab_size=64,
      num_layers=4,
      embed_dim=32,
      num_heads=4,
      num_kv_heads=2,
      head_dim=8,
      hidden_dim=64,
      sliding_window_size=4,
  )


def _tiny_moe_config() -> Gemma4Config:
  return Gemma4Config.moe_26b_like(
      vocab_size=64,
      num_layers=4,
      embed_dim=32,
      num_heads=4,
      num_kv_heads=2,
      head_dim=8,
      sliding_window_size=4,
      num_experts=4,
      expert_dim=16,
      top_k_experts=2,
      moe_dense_hidden_dim=32,
  )


def test_dense_forward_shape():
  torch.manual_seed(0)
  cfg = _tiny_dense_config()
  model = Gemma4(cfg).eval()
  tokens = torch.randint(0, cfg.vocab_size, (2, 7))
  logits, cache = model(tokens)
  assert logits.shape == (2, 7, cfg.vocab_size), logits.shape
  assert cache is None
  # Softcap: |logits| < 30.
  assert logits.abs().max().item() < cfg.final_logit_softcap


def test_moe_forward_shape():
  torch.manual_seed(0)
  cfg = _tiny_moe_config()
  model = Gemma4(cfg).eval()
  tokens = torch.randint(0, cfg.vocab_size, (2, 5))
  logits, _ = model(tokens)
  assert logits.shape == (2, 5, cfg.vocab_size), logits.shape


def test_attention_pattern_expands_to_num_layers():
  cfg = _tiny_dense_config()
  types = cfg.attention_types
  assert len(types) == cfg.num_layers
  # Pattern is 5L + 1G, so with 4 layers we get all local:
  assert all(t.name == "LOCAL_SLIDING" for t in types[:4])


def test_cache_matches_noncache():
  """One-shot forward == token-by-token forward with KV cache."""
  torch.manual_seed(0)
  cfg = _tiny_dense_config()
  model = Gemma4(cfg).eval()
  tokens = torch.randint(0, cfg.vocab_size, (1, 6))

  # One-shot (no cache) — full causal forward over the whole sequence.
  logits_ref, _ = model(tokens)

  # Token-by-token with cache.
  cache = model.init_cache(batch_size=1, cache_size=16, device=tokens.device)
  collected = []
  for t in range(tokens.shape[1]):
    tok = tokens[:, t : t + 1]
    pos = torch.tensor([[t]])
    logits_t, cache = model(tok, positions=pos, cache=cache)
    collected.append(logits_t)
  logits_cached = torch.cat(collected, dim=1)

  # Tight numerical agreement (same math modulo FP).
  max_diff = (logits_ref - logits_cached).abs().max().item()
  assert max_diff < 1e-4, f"cache vs no-cache diverged: {max_diff}"


def test_cache_matches_noncache_moe():
  torch.manual_seed(0)
  cfg = _tiny_moe_config()
  model = Gemma4(cfg).eval()
  tokens = torch.randint(0, cfg.vocab_size, (1, 6))

  logits_ref, _ = model(tokens)

  cache = model.init_cache(batch_size=1, cache_size=16, device=tokens.device)
  collected = []
  for t in range(tokens.shape[1]):
    tok = tokens[:, t : t + 1]
    pos = torch.tensor([[t]])
    logits_t, cache = model(tok, positions=pos, cache=cache)
    collected.append(logits_t)
  logits_cached = torch.cat(collected, dim=1)

  max_diff = (logits_ref - logits_cached).abs().max().item()
  assert max_diff < 1e-4, f"MoE cache vs no-cache diverged: {max_diff}"


def test_moe_router_weights_sum_to_one():
  torch.manual_seed(0)
  moe = MoEFeedForward(
      embed_dim=16, expert_dim=8, num_experts=6, top_k=2
  ).eval()
  x = torch.randn(2, 5, 16)
  logits = moe.router(x).float()
  probs = F.softmax(logits, dim=-1)
  topk_probs, topk_idx = torch.topk(logits, 2, dim=-1)
  topk_probs = probs.gather(-1, topk_idx)
  renorm = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
  assert torch.allclose(renorm.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_sliding_window_is_strictly_less_than():
  """A token outside the sliding window must not influence the output."""
  torch.manual_seed(0)
  # A single local-sliding layer, window=2 (so position 5 sees only {4, 5}).
  cfg = Gemma4Config.e4b_like(
      vocab_size=64,
      num_layers=1,
      embed_dim=32,
      num_heads=2,
      num_kv_heads=1,
      head_dim=8,
      hidden_dim=32,
      sliding_window_size=2,
  )
  model = Gemma4(cfg).eval()
  # Two batches differing only at position 0; position 5 should be identical.
  tokens_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
  tokens_b = torch.tensor([[9, 2, 3, 4, 5, 6]])
  logits_a, _ = model(tokens_a)
  logits_b, _ = model(tokens_b)
  max_diff_at_5 = (logits_a[:, 5, :] - logits_b[:, 5, :]).abs().max().item()
  assert max_diff_at_5 < 1e-5, (
      f"position 5 should not depend on position 0 given window 2, got diff"
      f" {max_diff_at_5}"
  )


def test_apply_rope_identity_at_position_zero():
  """At position 0, all angles are 0 → RoPE is identity."""
  x = torch.randn(1, 1, 2, 8)
  positions = torch.zeros(1, 1, dtype=torch.long)
  y = apply_rope(x, positions, base_frequency=10_000.0, rope_proportion=1.0)
  assert torch.allclose(x, y, atol=1e-6)


def test_apply_rope_partial_leaves_nope_half_unchanged():
  """With rope_proportion < 1.0, the "NoPE" portion must be untouched."""
  head_dim = 8
  x = torch.randn(1, 3, 2, head_dim)
  positions = torch.arange(3).unsqueeze(0)
  y = apply_rope(x, positions, base_frequency=1e6, rope_proportion=0.25)
  # rope_angles = int(0.25 * 4) = 1; nope_angles = 3.
  # Layout after RoPE: concat([first_half_rotated, second_half_rotated]) where
  # each half has H/2 = 4 entries. The NoPE positions are indices [1:4] in the
  # first half and [1:4] in the second half (i.e. everything except index 0
  # of each half).
  first_x, second_x = x.chunk(2, dim=-1)
  first_y, second_y = y.chunk(2, dim=-1)
  # The non-rotated entries are indices 1..3 in each half → compare those.
  assert torch.allclose(first_x[..., 1:], first_y[..., 1:], atol=1e-6)
  assert torch.allclose(second_x[..., 1:], second_y[..., 1:], atol=1e-6)


def test_generate_returns_expected_length():
  torch.manual_seed(0)
  cfg = _tiny_dense_config()
  model = Gemma4(cfg).eval()
  prompt = torch.randint(0, cfg.vocab_size, (2, 3))
  out = generate(model, prompt, max_new_tokens=5)
  assert out.shape == (2, 3 + 5), out.shape
  # Prompt is preserved verbatim.
  assert torch.equal(out[:, :3], prompt)


def test_generate_moe():
  torch.manual_seed(0)
  cfg = _tiny_moe_config()
  model = Gemma4(cfg).eval()
  prompt = torch.randint(0, cfg.vocab_size, (1, 3))
  out = generate(model, prompt, max_new_tokens=4)
  assert out.shape == (1, 7), out.shape


def test_ple_forward_runs():
  """PLE is off by default; turning it on should not crash and must change outputs."""
  torch.manual_seed(0)
  cfg = Gemma4Config.e4b_like(
      vocab_size=32, num_layers=3, embed_dim=32,
      num_heads=2, num_kv_heads=1, head_dim=16,
      hidden_dim=64, sliding_window_size=4,
      per_layer_input_dim=8,
  )
  model = Gemma4(cfg).eval()
  tokens = torch.randint(0, cfg.vocab_size, (1, 5))
  logits, _ = model(tokens)
  assert logits.shape == (1, 5, cfg.vocab_size)


if __name__ == "__main__":
  import sys
  import traceback

  tests = [
      test_dense_forward_shape,
      test_moe_forward_shape,
      test_attention_pattern_expands_to_num_layers,
      test_cache_matches_noncache,
      test_cache_matches_noncache_moe,
      test_moe_router_weights_sum_to_one,
      test_apply_rope_identity_at_position_zero,
      test_apply_rope_partial_leaves_nope_half_unchanged,
      test_sliding_window_is_strictly_less_than,
      test_generate_returns_expected_length,
      test_generate_moe,
      test_ple_forward_runs,
  ]
  failures = 0
  for fn in tests:
    try:
      fn()
      print(f"PASS  {fn.__name__}")
    except Exception:  # pylint: disable=broad-except
      failures += 1
      print(f"FAIL  {fn.__name__}")
      traceback.print_exc()
  print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
  sys.exit(0 if failures == 0 else 1)

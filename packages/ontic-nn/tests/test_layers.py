"""Tests for ontic_nn.layers."""

import torch
import torch.nn.functional as F

from ontic_nn.layers import Attention, Block, DropPath, SwiGLUFFNFused


def _manual_attention(attn: Attention, x: torch.Tensor) -> torch.Tensor:
    b, n, c = x.shape
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = (q @ k.transpose(-2, -1)) * attn.scale
    out = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c)
    return attn.proj(out)


def test_block_forward_shape():
    torch.manual_seed(0)
    block = Block(dim=32, num_heads=4, init_values=1.0, drop_path=0.1).eval()
    x = torch.randn(2, 9, 32)
    assert block(x).shape == (2, 9, 32)


def test_block_swiglu_and_cross_attention():
    torch.manual_seed(0)
    block = Block(dim=32, num_heads=2, ffn_layer=SwiGLUFFNFused).eval()
    x = torch.randn(2, 5, 32)
    context = torch.randn(2, 7, 32)
    assert block(x, context=context).shape == (2, 5, 32)


def test_drop_path_identity_in_eval():
    dp = DropPath(0.5).eval()
    x = torch.randn(4, 3, 8)
    assert torch.equal(dp(x), x)
    dp.train()
    y = dp(x)
    dropped = (y.flatten(1) == 0).all(dim=1)
    kept = ~dropped
    assert torch.allclose(y[kept], x[kept] / 0.5)


def test_sdpa_attention_matches_manual_softmax():
    torch.manual_seed(0)
    attn = Attention(dim=32, num_heads=4, qkv_bias=True).eval()
    x = torch.randn(2, 11, 32)
    with torch.no_grad():
        assert torch.allclose(attn(x), _manual_attention(attn, x), atol=1e-5)


def test_cross_attention_matches_self_attention_on_same_context():
    torch.manual_seed(0)
    attn = Attention(dim=16, num_heads=2, qkv_bias=True).eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        assert torch.allclose(attn(x, context=x), attn(x), atol=1e-6)
        ctx = torch.randn(1, 4, 16)
        q = F.linear(x, attn.qkv.weight[:16], attn.qkv.bias[:16])
        kv = F.linear(ctx, attn.qkv.weight[16:], attn.qkv.bias[16:])
        k, v = kv.chunk(2, dim=-1)
        split = lambda t: t.reshape(1, -1, 2, 8).transpose(1, 2)  # noqa: E731
        scores = (split(q) @ split(k).transpose(-2, -1)) * attn.scale
        ref = attn.proj((scores.softmax(-1) @ split(v)).transpose(1, 2).reshape(1, 6, 16))
        assert torch.allclose(attn(x, context=ctx), ref, atol=1e-5)

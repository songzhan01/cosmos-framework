#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""NDHWC streaming CausalConv3d + ResidualBlock: validate correctness vs the
original NCDHW streaming path on a 2-chunk stream (prime 1-frame + 4-frame chunk).
This is the core de-risk for the full NDHWC encoder (the streaming cache must
thread NDHWC). If correct, the full encoder assembly follows mechanically."""
import torch, torch.nn.functional as F
from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import (
    CausalConv3d, RMS_norm, ResidualBlock, CACHE_T, _contiguous_clone, _update_cache_and_apply)

# The CUTLASS op (torch.ops.fp16acc.*) is registered by vae_install._load_op()
# — either from the prebuilt .so shipped in the package or via JIT fallback.
# We ensure registration lazily so any entry path (install() or a dev run())
# works without a module-level JIT compile.
_op_ready = False
def _ensure_op():
    global _op_ready
    if not _op_ready:
        from . import vae_install
        vae_install._load_op()
        _op_ready = True


# Pre-convert a CausalConv3d's weight to KTRSC fp16 (cached on the module).
# Intentionally a plain module attribute (NOT register_buffer): this is a
# derived transient cache on a frozen eval module, not model state. A buffer
# would couple it to Module._apply (dtype/device transforms), risking an
# accidental cast of the fp16 cache back to bf16, which breaks the fp16-only
# CUTLASS op. hasattr() guards re-derivation.
def _w_ktrsc(layer):
    if not hasattr(layer, "_w_ktrsc"):
        layer._w_ktrsc = layer.weight.permute(0, 2, 3, 4, 1).contiguous().to(torch.float16)
    return layer._w_ktrsc

def causal_conv3d_ndhwc(layer, x, cache_x=None):
    """NDHWC [B,T,H,W,C] -> [B,T',H',W',C']. Mirrors CausalConv3d.forward but NDHWC.
    Adds bias (the original nn.Conv3d.forward adds it; CUTLASS path must too).
    If x is already fp16, no cast (saves ~150ms across 62 convs at 544x736).

    Optimization: spatial pad H/W FOLDED into CUTLASS's analytic iterator
    (eliminates F.pad copy). Causal-T (asymmetric, left-only) still uses F.pad.
    The 1.8ms F.pad on stage0 input (1,17,272,368,160) gets reduced to a small
    T-only pad of 1.5ms -> ~1ms savings per conv * ~30 convs."""
    _ensure_op()
    pt, ph, pw = layer._padding[4] // 2, layer._padding[2], layer._padding[0]
    if pt == 0 and ph == 0 and pw == 0 and cache_x is None:
        x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        st, sh, sw = layer.stride
        out = torch.ops.fp16acc.cutlass_conv3d(x_fp16, _w_ktrsc(layer), int(st), int(sh), int(sw))
        if layer.bias is not None:
            if not hasattr(layer, "_b_ndhwc"):
                layer._b_ndhwc = layer.bias.view(1, 1, 1, 1, -1).to(torch.float16)
            out = out + layer._b_ndhwc
        return out

    if cache_x is not None and 2 * pt > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=1)
        rem_t = 2 * pt - cache_x.shape[1]
    else:
        rem_t = 2 * pt
    if rem_t > 0:
        # T-only causal pad (asymmetric). Smaller than full HW pad.
        x = F.pad(x, (0, 0, 0, 0, 0, 0, rem_t, 0))
    st, sh, sw = layer.stride
    x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)
    if ph == 0 and pw == 0:
        out = torch.ops.fp16acc.cutlass_conv3d(x_fp16, _w_ktrsc(layer), int(st), int(sh), int(sw))
    else:
        out = torch.ops.fp16acc.cutlass_conv3d_padded(
            x_fp16, _w_ktrsc(layer), int(st), int(sh), int(sw),
            0, int(ph), int(pw))
    if layer.bias is not None:
        if not hasattr(layer, "_b_ndhwc"):
            layer._b_ndhwc = layer.bias.view(1, 1, 1, 1, -1).to(torch.float16)
        out = out + layer._b_ndhwc
    return out

def _update_cache_and_apply_ndhwc(x, layer, feat_cache, feat_idx):
    idx = feat_idx[0]
    # cache_x is the last CACHE_T (=2) frames; if x has only 1 frame, prepend prev cache's last frame.
    # Slicing creates a view; clone to a contiguous fp16 NDHWC buffer (matches CACHE_T frames).
    # NOTE: dropping the .clone() (just a view) is unsafe because the underlying buffer of x
    # is overwritten by the next conv's output (CUTLASS writes in-place via update()).
    cache_x = x[:, -CACHE_T:, :, :, :].contiguous()
    if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat([feat_cache[idx][:, -1:, :, :, :], cache_x], dim=1)
    x = causal_conv3d_ndhwc(layer, x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x

def rms_norm_ndhwc(layer, x):  # x [B,T,H,W,C]; channel_last -> dim=-1
    # Cache fp16 gamma so we don't keep recomputing it.
    if not hasattr(layer, "_g16"):
        layer._g16 = layer.gamma.reshape(-1).to(torch.float16)
    # F.normalize keeps dtype; multiplying by fp16 gamma keeps fp16. bias is 0 in this VAE.
    return F.normalize(x, dim=-1) * (layer.scale * layer._g16)

def residual_block_ndhwc(self, x, feat_cache=None, feat_idx=[0]):
    """NDHWC ResidualBlock. x [B,T,H,W,C]. Mirrors ResidualBlock.forward."""
    # Keep fp16 throughout (no cast back to caller's dtype between sublayers).
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if isinstance(self.shortcut, CausalConv3d):
        h = causal_conv3d_ndhwc(self.shortcut, x)
    else:
        h = x
    for layer in self.residual:
        if isinstance(layer, CausalConv3d):
            if feat_cache is not None:
                x = _update_cache_and_apply_ndhwc(x, layer, feat_cache, feat_idx)
            else:
                x = causal_conv3d_ndhwc(layer, x)
        elif isinstance(layer, RMS_norm):
            x = rms_norm_ndhwc(layer, x)
        else:  # SiLU, Dropout (operate on fp16 directly)
            x = layer(x)
    return x + h

def avg_down_3d_ndhwc(self, x):  # x [B,T,H,W,C] -> [B,T//ft,H//fs,W//fs,out]
    # AvgPool downsample shortcut (Resample.avg_shortcut). NDHWC-native: fold
    # the ft/fs/factor grouping into a mean over group_size (no copy, no cuDNN).
    ft, fs = self.factor_t, self.factor_s
    pad_t = (ft - x.shape[1] % ft) % ft
    x = F.pad(x, (0, 0, 0, 0, 0, 0, pad_t, 0))  # pad T (dim1) right
    B, T, H, W, C = x.shape
    x = x.view(B, T // ft, ft, H // fs, fs, W // fs, fs, C)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()  # [B,T//ft,H//fs,W//fs,C,ft,fs,fs]
    x = x.view(B, T // ft, H // fs, W // fs, C * self.factor)
    x = x.view(B, T // ft, H // fs, W // fs, self.out_channels, self.group_size)
    return x.mean(dim=5)  # over group_size -> [B,T//ft,H//fs,W//fs,out]

def attention_ndhwc(self, x):  # x [B,T,H,W,C] fp16 -> channels_last 2D Conv2d
    # AttentionBlock. View NDHWC as channels_last NCHW (stride trick) so the
    # 2D Conv2d / SDPA run on NHWC cuDNN natively (no NDHWC->NCHW copy).
    b, t, h, w, c = x.shape
    identity = x
    x_cl = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2).to(torch.bfloat16)
    x_cl = self.norm(x_cl)
    qkv = self.to_qkv(x_cl)
    q, k, v = qkv.reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.squeeze(1).permute(0, 2, 1).contiguous().reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
    o = self.proj(o)
    o = o.permute(0, 2, 3, 1).reshape(b, t, h, w, c).contiguous().to(torch.float16)
    return o + identity

# ---- validation: streaming 2-chunk vs original NCDHW ----
def run():
    torch.manual_seed(0)
    C_in, C_out = 160, 160
    rb = ResidualBlock(C_in, C_out, 0.0).to("cuda").to(torch.bfloat16).eval()
    # prime chunk: 1 frame; steady chunk: 4 frames. spatial 32x32.
    T_p, T_c, H, W = 1, 4, 32, 32
    x_prime = torch.randn(1, C_in, T_p, H, W, device="cuda", dtype=torch.bfloat16) * 0.1
    x_chunk = torch.randn(1, C_in, T_c, H, W, device="cuda", dtype=torch.bfloat16) * 0.1

    # original NCDHW streaming (feat_idx resets per chunk, cache shared across chunks)
    n_conv = sum(1 for m in rb.modules() if isinstance(m, CausalConv3d))
    cache_orig = [None] * n_conv
    with torch.no_grad():
        o1 = rb(x_prime, cache_orig, [0])
        o2 = rb(x_chunk, cache_orig, [0])
    ref = torch.cat([o1, o2], dim=2).float()  # NCDHW [1,C_out,5,H,W]

    # NDHWC streaming
    cache_nd = [None] * n_conv
    with torch.no_grad():
        xp = x_prime.permute(0, 2, 3, 4, 1).contiguous()
        xc = x_chunk.permute(0, 2, 3, 4, 1).contiguous()
        n1 = residual_block_ndhwc(rb, xp, cache_nd, [0])
        n2 = residual_block_ndhwc(rb, xc, cache_nd, [0])
    out = torch.cat([n1, n2], dim=1).permute(0, 4, 1, 2, 3).contiguous().float()  # NCDHW

    err = (out - ref).abs()
    rel = err / (ref.abs() + 1e-3)
    print(f"NDHWC streaming ResidualBlock vs NCDHW: max_abs={err.max().item():.5f} "
          f"mean_rel={rel.mean().item():.4f} ref_max={ref.abs().max().item():.4f} "
          f"{'PASS' if torch.allclose(out, ref, atol=0.05, rtol=0.05) else 'FAIL'}")

if __name__ == "__main__":
    run()

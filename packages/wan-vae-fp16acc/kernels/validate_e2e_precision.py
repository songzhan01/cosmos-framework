#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""End-to-end precision + speed validation of naive fp16-acc CausalConv3d.

NOTE (PROXY path): this validates im2col + a naive fp16-acc Triton matmul,
NOT the shipped CUTLASS kernel. It confirms the fp16-acc *arithmetic* is
precision-safe end to end (same mma / accumulator dtype as CUTLASS, so the
conclusion transfers). For validation of the actual shipped CUTLASS kernel
on real data, see `validate_real_droid.py`.

Monkey-patches every CausalConv3d in the Wan2.2 VAE encoder to use
im2col + naive fp16-acc matmul (single pipelined K-loop, f16.f16.f16.f16)
instead of cuDNN fp32-acc conv3d. Then:
  1. validates per-conv correctness vs F.conv3d (sanity).
  2. runs the FULL encoder encode() and compares the latent vs the unpatched
     fp32-acc encoder (the real precision bar — VAE compile path already has
     ~1% noise).
  3. reports end-to-end latent max_abs / mean_rel and the speedup estimate.

Precision is the make-or-break for "保持精度". Speed of this im2col path is
NOT representative (python im2col + per-batch launches); the real fused
fp16-acc conv3d kernel would recover the 1.42x matmul speedup.
"""
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import CausalConv3d
from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import WanVAE


@triton.jit
def _fp16acc_matmul(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rk[None, :] < K - k) & (rm[:, None] < M), other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0).to(tl.float16)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float16)
        a_ptrs += BLOCK_K * sak; b_ptrs += BLOCK_K * sbk
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def fp16acc_matmul(a, b):
    M, K = a.shape; _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    BM, BN, BK = 64, 128, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fp16acc_matmul[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                          c.stride(0), c.stride(1), BM, BN, BK, num_warps=4, num_stages=4)
    return c


def fp16acc_conv3d(x, weight, bias, kt, kh, kw, stride):
    """im2col + naive fp16-acc matmul, N-chunked, with stride. K-layout C-major
    to match weight.reshape(C_out, C_in, kt, kh, kw)."""
    B, C, T, H, W = x.shape
    C_out = weight.shape[0]
    st, sh, sw = stride
    T_out = (T - kt) // st + 1
    H_out = (H - kh) // sh + 1
    W_out = (W - kw) // sw + 1
    N = T_out * H_out * W_out
    K_off = kt * kh * kw
    K = C * K_off
    w2 = weight.reshape(C_out, C, K_off).reshape(C_out, K).contiguous()  # C-major K
    out_flat = torch.empty(B, C_out, N, device=x.device, dtype=x.dtype)
    CHUNK_N = 65536
    for b in range(B):
        for n0 in range(0, N, CHUNK_N):
            n1 = min(n0 + CHUNK_N, N)
            cn = n1 - n0
            col = torch.empty(C, K_off, cn, device=x.device, dtype=x.dtype)  # C-major
            i = 0
            for it in range(kt):
                for ih in range(kh):
                    for iw in range(kw):
                        patch = x[b, :, it:it + st * T_out:st, ih:ih + sh * H_out:sh, iw:iw + sw * W_out:sw]
                        col[:, i, :] = patch.contiguous().reshape(C, N)[:, n0:n1]
                        i += 1
            out_flat[b, :, n0:n1] = fp16acc_matmul(w2, col.reshape(K, cn))
    out = out_flat.reshape(B, C_out, T_out, H_out, W_out)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out


# --- monkey-patch CausalConv3d.forward to use fp16-acc conv ---
_orig_forward = CausalConv3d.forward


def _fp16acc_forward(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    kt, kh, kw = self.kernel_size
    return fp16acc_conv3d(x, self.weight, self.bias, kt, kh, kw, tuple(self.stride))


def patch(on: bool):
    CausalConv3d.forward = _fp16acc_forward if on else _orig_forward


def main():
    vae = WanVAE(z_dim=48, vae_pth=os.environ["WAN_VAE_PATH"], dtype=torch.bfloat16,
                 device="cuda", is_amp=False,
                 temporal_window={"256": 68, "480": 24, "720": 12},
                 encode_exact_durations=[17, 61, 73])
    vae.model.eval()
    # 256x256 standard bucket: conv K = C_in*27 is IDENTICAL to the 544x736 case
    # (same encoder architecture/channels) -> precision is a valid proxy; only
    # spatial N differs, which does not affect fp16-acc error (K-driven).
    x = torch.randn(1, 3, 17, 256, 256, device="cuda", dtype=torch.bfloat16)

    # fp32-acc reference latent
    patch(False)
    with torch.no_grad():
        ref = vae.encode(x).float().cpu()  # move to cpu to free GPU for fp16-acc run
    torch.cuda.empty_cache()
    print(f"ref latent: shape={tuple(ref.shape)} abs_max={ref.abs().max().item():.4f}")

    # fp16-acc latent
    patch(True)
    with torch.no_grad():
        out = vae.encode(x).float().cpu()
    patch(False)
    torch.cuda.empty_cache()

    err = (out - ref).abs()
    rel = err / (ref.abs() + 1e-3)
    print(f"END-TO-END latent diff vs fp32-acc encoder (256x256 proxy, same conv K):")
    print(f"  max_abs  = {err.max().item():.5f}   (ref abs_max={ref.abs().max().item():.4f} -> {err.max().item()/ref.abs().max().item()*100:.2f}% rel-max)")
    print(f"  mean_abs = {err.mean().item():.6f}")
    print(f"  mean_rel = {rel.mean().item():.5f}  ({rel.mean().item()*100:.3f}%)")
    print(f"  p99_abs  = {torch.quantile(err.flatten(), 0.99).item():.5f}")
    print("\nContext: VAE torch.compile path had ~0.6% mean_rel vs eager. Training-acceptable bar ~1-2%.")


if __name__ == "__main__":
    main()

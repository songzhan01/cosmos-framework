#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""Correctness + speed of CUTLASS fp16-acc conv3d vs F.conv3d (fp32-acc cuDNN).

Uses the shipped package op (torch.ops.fp16acc.*) — registered via the
package's _load_op(), which respects the in-package prebuilt .so (no JIT, no
hardcoded CUTLASS header path).
"""
import torch
from wan_vae_fp16acc.vae_install import _load_op

_load_op()  # register torch.ops.fp16acc.* without patching the encoder


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


for name, (N, C, D, H, W, Kout) in [
    ("stage2 640ch", (1, 640, 5, 68, 92, 640)),
    ("stage1 320ch", (1, 320, 9, 136, 184, 320)),
    ("stage0 160ch", (1, 160, 17, 272, 368, 160)),
]:
    torch.manual_seed(0)
    x_ncdhw = torch.randn(N, C, D, H, W, device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn(Kout, C, 3, 3, 3, device="cuda", dtype=torch.float16) * (1.0 / (C * 27) ** 0.5)
    ref = torch.nn.functional.conv3d(x_ncdhw, w, stride=1, padding=1)   # fp32-acc, [N,Kout,D,H,W]

    # valid-conv flow: pre-pad symmetric 1, then CUTLASS pad=0 on padded input
    xp = torch.nn.functional.pad(x_ncdhw, (1, 1, 1, 1, 1, 1))           # [N,C,D+2,H+2,W+2]
    A = xp.permute(0, 2, 3, 4, 1).contiguous()                          # [N,D+2,H+2,W+2,C]
    B = w.permute(0, 2, 3, 4, 1).contiguous()                           # [Kout,3,3,3,C]
    out_ndhwc = torch.ops.fp16acc.cutlass_conv3d(A, B, 1, 1, 1)         # [N,D,H,W,Kout]
    out = out_ndhwc.permute(0, 4, 1, 2, 3).contiguous()                 # [N,Kout,D,H,W]

    err = (out.float() - ref.float()).abs()
    rel = err / (ref.float().abs() + 1e-3)
    ok = torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05)
    ms_c = bench(lambda: torch.ops.fp16acc.cutlass_conv3d(A, B, 1, 1, 1))
    ms_r = bench(lambda: torch.nn.functional.conv3d(x_ncdhw, w, stride=1, padding=1))
    flops = 2 * N * Kout * C * 27 * D * H * W
    tf_c = flops / (ms_c / 1000) / 1e12; tf_r = flops / (ms_r / 1000) / 1e12
    print(f"{name:16s} max_abs={err.max().item():.5f} mean_rel={rel.mean().item():.4f} "
          f"{'PASS' if ok else 'FAIL'} | cutlass {ms_c:.2f}ms/{tf_c:.0f}T  cuDNN {ms_r:.2f}ms/{tf_r:.0f}T  speedup {tf_c/tf_r:.2f}x")

# stride=2 (time_conv) correctness
torch.manual_seed(0)
x = torch.randn(1, 64, 9, 40, 40, device="cuda", dtype=torch.float16) * 0.1
w = torch.randn(64, 64, 3, 1, 1, device="cuda", dtype=torch.float16) * 0.05
ref = torch.nn.functional.conv3d(x, w, stride=(2, 1, 1), padding=(1, 0, 0))
xp = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1))                    # pad T only (1 each side)
A = xp.permute(0, 2, 3, 4, 1).contiguous()
B = w.permute(0, 2, 3, 4, 1).contiguous()
out = torch.ops.fp16acc.cutlass_conv3d(A, B, 2, 1, 1).permute(0, 4, 1, 2, 3).contiguous()
err = (out.float() - ref.float()).abs()
print(f"stride(2,1,1) time_conv: max_abs={err.max().item():.5f}  "
      f"{'PASS' if torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05) else 'FAIL'}")

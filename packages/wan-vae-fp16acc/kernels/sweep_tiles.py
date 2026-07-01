"""Sweep per-stage CUTLASS tile to find optimum per shape.

Default heuristic in cutlass_conv3d_ext.cu::select_tile():
  K=160 (stage0) -> 128x256,  K=320 (stage1) -> 256x128,  K=640 (stage2) -> 256x128

Uses the shipped package op (torch.ops.fp16acc.*), incl. set_tile_overrides,
so no JIT build / hardcoded CUTLASS path is needed.

Shapes are the DOMINANT full-chunk conv3d shapes a DROID training encode feeds
to the op on 4090 (captured via capture_droid_shapes.py from a real
[1,3,33,544,736] encode with temporal_window={"480":24}, encode_exact_durations=[33]).
The encoder streams prime(1) + 24-chunk + 8-remainder; the 24-chunk (12 at the
deepest stage after a temporal downsample) dominates runtime, so we tune for it.
H/W (272x368 etc.) have spatial pad folded into CUTLASS's analytic iterator.

Usage:
  python sweep_tiles.py                          # quick: 1 round x 50 iters
  python sweep_tiles.py --rounds 3 --iters 200   # rigorous (noise-floor confirmation)
"""
import argparse
import statistics

import torch

from wan_vae_fp16acc.vae_install import _load_op

_load_op()

TILE_NAMES = ["128x128", "128x256", "256x128", "64x256"]

# DROID training shapes (captured on 4090, [1,3,33,544,736] input, tw=24):
#   stage0 (Cout=160): full 24-frame chunk at 272x368
#   stage1 (Cout=320): full 24-frame chunk at 136x184
#   stage2 (Cout=640): full 12-frame chunk at 68x92 (T halved by resample time_conv)
# default_tile = the value baked into cutlass_conv3d_ext.cu::select_tile(), used
# only for the [keep/CHANGE] verdict (does not affect the sweep itself).
shapes = [
    ("stage0 DROID K=160", 1, 160, 24, 272, 368, 160, 1),  # default 128x256
    ("stage1 DROID K=320", 1, 320, 24, 136, 184, 320, 2),  # default 256x128
    ("stage2 DROID K=640", 1, 640, 12,  68,  92, 640, 2),  # default 256x128
]


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument(
        "--rounds", type=int, default=1,
        help="runs per tile; take median (use 3+ for noise-floor cases)",
    )
    args = ap.parse_args()

    for name, N, Cin, T, H, W, Cout, default_tile in shapes:
        A = torch.randn(N, T, H, W, Cin, device="cuda", dtype=torch.float16) * 0.1
        Bw = torch.randn(Cout, 3, 3, 3, Cin, device="cuda", dtype=torch.float16) * 0.01
        flops = 2 * N * Cout * Cin * 27 * (T - 2) * (H - 2) * (W - 2)
        fn = lambda: torch.ops.fp16acc.cutlass_conv3d(A, Bw, 1, 1, 1)
        print(f"\n{name}  (default: tile {default_tile} {TILE_NAMES[default_tile]}):")
        best = None
        for tile in range(4):
            # Set tile for the matching stage only; the other two stay at default (-1).
            s0 = tile if Cout == 160 else -1
            s1 = tile if Cout == 320 else -1
            s2 = tile if Cout == 640 else -1
            torch.ops.fp16acc.set_tile_overrides(s0, s1, s2)
            runs = [bench(fn, args.iters, args.warmup) for _ in range(args.rounds)]
            ms = runs[0] if len(runs) == 1 else statistics.median(runs)
            tf = flops / (ms / 1000) / 1e12
            mark = "*" if best is None or ms < best[1] else " "
            dflt = " (default)" if tile == default_tile else ""
            runs_str = f"  runs={[round(r, 4) for r in runs]}" if args.rounds > 1 else ""
            print(f"  [{mark}] tile {tile} {TILE_NAMES[tile]:8s} {ms:7.4f}ms  {tf:6.1f}T{dflt}{runs_str}")
            if best is None or ms < best[1]:
                best = (tile, ms, tf)
        verdict = "CHANGE" if best[0] != default_tile else "keep"
        print(f"  -> best: tile {best[0]} {TILE_NAMES[best[0]]} = {best[1]:.4f}ms / {best[2]:.1f}T  [{verdict}]")
    torch.ops.fp16acc.set_tile_overrides(-1, -1, -1)  # revert to baked defaults


if __name__ == "__main__":
    main()

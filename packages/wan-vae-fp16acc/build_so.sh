#!/usr/bin/env bash
# Build the CUTLASS fp16-acc conv3d extension as a prebuilt .so.
# The .so is shipped inside the package (wan_vae_fp16acc/lib/) so that after
# `pip install .`, importing the package + setting COSMOS_VAE_FP16ACC=1 loads
# the op with ZERO JIT startup cost.
#
# We delegate compilation to torch.utils.cpp_extension.load() (same machinery as
# the JIT fallback) so torch/c10/ATen headers + link libs are resolved
# automatically. Hand-rolling nvcc flags for a torch C++ extension is fragile
# (needs torch include dirs, python include dir, -lc10 -ltorch ...). The
# produced .so is copied into wan_vae_fp16acc/lib/.
#
# CUTLASS source: we use upstream CUTLASS UNMODIFIED (no local patch). To make
# the build reproducible, a specific CUTLASS version is pinned (CUTLASS_PIN
# below). If CUTLASS_PATH is set, that checkout is used as-is (you must ensure
# it matches the pin for reproducibility). If unset, the pinned version is
# git-cloned once into a cache dir and reused.
#
# Requirements: a venv with torch + nvcc + (auto-cloned CUTLASS, or CUTLASS_PATH).
# Targets the current device's arch automatically (sm_89 on a 4090).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pinned CUTLASS version — verified to build the 1.45x kernel on sm_89.
# v4.4.0-11-gb9847690 (upstream, unmodified).
CUTLASS_PIN="b9847690c5838ac3d909ebc163ed16c388802485"
CUTLASS_REPO="https://github.com/NVIDIA/cutlass.git"
CACHE_DIR="${WAN_VAE_FP16ACC_CACHE:-$HOME/.cache/wan-vae-fp16acc}/cutlass"

if [[ -z "${CUTLASS_PATH:-}" ]]; then
    if [[ ! -d "$CACHE_DIR/.git" ]]; then
        echo "[build_so] CUTLASS_PATH unset; cloning CUTLASS @ ${CUTLASS_PIN:0:8} -> $CACHE_DIR"
        if ! git clone --quiet "$CUTLASS_REPO" "$CACHE_DIR"; then
            echo "ERROR: git clone failed. If network is blocked, set http_proxy/https_proxy" >&2
            echo "       (e.g. export https_proxy=http://192.168.112.80:18000) or point" >&2
            echo "       CUTLASS_PATH at a local CUTLASS checkout." >&2
            exit 1
        fi
    fi
    git -C "$CACHE_DIR" fetch --quiet origin "$CUTLASS_PIN" 2>/dev/null || \
        git -C "$CACHE_DIR" fetch --quiet --depth 1 origin "$CUTLASS_PIN" 2>/dev/null || true
    git -C "$CACHE_DIR" checkout --quiet "$CUTLASS_PIN"
    CUTLASS_PATH="$CACHE_DIR"
    echo "[build_so] using pinned CUTLASS @ ${CUTLASS_PIN:0:8} ($CACHE_DIR)"
else
    echo "[build_so] using CUTLASS_PATH=$CUTLASS_PATH (pin $CUTLASS_PIN not enforced)"
fi
export CUTLASS="$CUTLASS_PATH"
mkdir -p wan_vae_fp16acc/lib

# CUTLASS template instantiation writes GB-scale intermediate .c files via nvcc.
# Default /tmp is often a small tmpfs/partition that fills up. Redirect nvcc's
# temp dir to a spacious location (override with TMPDIR env).
export TMPDIR="${TMPDIR:-$HOME/.cache/wan-vae-fp16acc/tmp}"
mkdir -p "$TMPDIR"

echo "[build_so] arch auto-detected by torch  TMPDIR=$TMPDIR"

python - <<'EOF'
import os, shutil
from torch.utils.cpp_extension import load
cutlass = os.environ["CUTLASS"]
ext = load(
    name="cutlass_conv3d_fp16acc_ext",
    sources=["wan_vae_fp16acc/cutlass_conv3d_ext.cu"],
    extra_include_paths=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
    extra_cuda_cflags=["-O3", "-std=c++17", "-DCUTLASS_ARCH_MMA_SM80_SUPPORTED"],
    verbose=False,
)
src = ext.__file__                      # .so in the JIT cache
dst = "wan_vae_fp16acc/lib/cutlass_conv3d_fp16acc_ext.so"
shutil.copy(src, dst)
print(f"[build_so] copied {src}")
print(f"       -> {dst}  ({os.path.getsize(dst)//1024} KB)")
EOF

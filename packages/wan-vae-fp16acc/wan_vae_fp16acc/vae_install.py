"""fp16-acc VAE encoder install hook.

Activation (auto): the package installs a .pth site-hook; setting
  export COSMOS_VAE_FP16ACC=1
triggers bootstrap() at interpreter start, which calls install() before the
training framework imports anything. No PYTHONPATH needed.

Activation (explicit):
  import wan_vae_fp16acc
  wan_vae_fp16acc.install()

What install() does when enabled:
  1. sm_capability check: only patches on consumer GeForce (sm_75/86/89, sm_120
     GeForce). Silently no-ops on data-center / workstation (Pro6000 sm_120:
     fp16-acc = fp32-acc = 500T, no gain).
  2. Loads the CUTLASS extension: prefers the prebuilt .so shipped in the
     package (wan_vae_fp16acc/lib/*.so, zero startup cost); falls back to JIT
     compile (~90s) when no .so is present.
  3. Wraps CUTLASS conv3d as `fp16acc::cutlass_conv3d(_padded)` custom op so
     torch.compile treats it as a black box (no graph break, pointwise around
     it stays fused). Meta impl is registered in C++ (TORCH_LIBRARY_IMPL Meta).
  4. Monkey-patches `Encoder3d.forward` with the NDHWC streaming impl.
     Class-level patch (persists until process exit); use scoped_install() to
     restrict to a single encode() call.

Constraints:
  - `torch.compile(mode="max-autotune-no-cudagraphs")` — cudagraphs conflicts
    with the op's per-shape kernel cache.
  - VAE weights are read-only (eval mode). Live-training the VAE with this
    patch is unsupported.
"""
import os
import torch
from contextlib import contextmanager

from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import (
    WanVAE, Encoder3d, CausalConv3d, RMS_norm, ResidualBlock, Resample, AttentionBlock,
    _update_cache_and_apply, _contiguous_clone, CACHE_T,
)


def _rank0() -> bool:
    """Whether this process is the rank-0 worker (or single-proc)."""
    return int(os.environ.get("RANK", "0")) == 0


def _log(msg: str) -> None:
    if _rank0():
        print(f"[fp16acc] {msg}", flush=True)


def _is_supported_arch() -> bool:
    """fp16-acc 2x lever: NVIDIA product segmentation — consumer GeForce RTX
    cards have fp32-acc pipe throttled to half of fp16-acc, so fp16-acc gives
    2x uplift. Data-center / workstation cards have fp32-acc at full speed
    (fp16-acc = fp32-acc, no gain).

    ALLOW (2x uplift confirmed):
      - sm_75 (RTX 20-series, T4): 130T fp16-acc / 65T fp32-acc = 2x
      - sm_86 (RTX 30-series, RTX A6000): 142T / 71T = 2x
      - sm_89 (RTX 4090): 330T / 165T = 2x  <- our validation target
      - sm_120 GeForce RTX 5090: 838T / ~419T = 2x (Blackwell consumer)

    DENY (no uplift):
      - sm_80 A100: 312T fp16-acc = 312T fp32-acc (data-center, unified)
      - sm_90 H100: 989T = 989T (data-center)
      - sm_100 B200: 2250T = 2250T (data-center)
      - sm_120 RTX PRO 6000 Blackwell: 500T = 500T (workstation, confirmed)

    sm_120 is ambiguous by capability alone — we also check device name to
    distinguish GeForce (2x) from PRO/workstation (1x).

    NB: tile heuristics are tuned only for sm_89 (RTX 4090); older archs will
    work but perf may not be optimal. Emit a warning when running on non-sm_89.
    """
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)

    # sm_120 ambiguity: GeForce RTX 5090 is 2x, RTX PRO 6000 is 1x, both cap 12.0
    if cap == (12, 0):
        if "GeForce" in name or "RTX 5090" in name:
            _log(f"WARNING: sm_120 GeForce ({name}) — CUTLASS tiles were tuned for sm_89, may need retuning")
            return True
        # PRO / workstation Blackwell: no fp16-acc uplift
        _log(f"skipping install: {name} (sm_120 workstation, fp16-acc = fp32-acc, no gain)")
        return False

    # Consumer / lower-end cards with confirmed 2x uplift
    supported = {(7, 5), (8, 6), (8, 9)}
    if cap in supported:
        if cap != (8, 9):
            _log(f"WARNING: sm_{cap[0]}{cap[1]} ({name}) — tiles tuned for sm_89, may need retuning")
        return True

    # A100 (8, 0) has 2x uplift but is data-center — allowing it is a policy call.
    # Conservative: deny to prevent surprising users. Set FP16ACC_ALLOW_A100=1 to override.
    if cap == (8, 0) and os.environ.get("FP16ACC_ALLOW_A100", "0") in ("1", "true", "yes"):
        _log(f"A100 override: enabling fp16-acc (2x uplift exists on A100 but path is untested)")
        return True

    _log(f"skipping install: {name} (sm_{cap[0]}{cap[1]}, no fp16-acc uplift or untested)")
    return False


# Public alias (re-exported by wan_vae_fp16acc.__init__).
is_supported_arch = _is_supported_arch


def _build_extension():
    """JIT-compile the CUTLASS extension (fallback when no prebuilt .so).

    Returns the loaded pybind module and registers torch.ops.fp16acc.*.
    Targets the CURRENT device's arch (not hardcoded) so the arch gate's other
    allowed archs (sm_75/86/120) get native SASS instead of PTX-JIT fallback.
    """
    from torch.utils.cpp_extension import load
    cutlass = os.environ.get("CUTLASS_PATH")
    if not cutlass:
        raise RuntimeError(
            "CUTLASS_PATH env var must point to a CUTLASS repo checkout "
            "(headers under $CUTLASS_PATH/include). Clone from "
            "https://github.com/NVIDIA/cutlass. (The prebuilt .so path does "
            "not need this; only the JIT fallback does.)"
        )
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "cutlass_conv3d_ext.cu")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    return load(
        name="cutlass_conv3d_fp16acc_ext",
        sources=[src],
        extra_include_paths=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
        extra_cuda_cflags=[f"-arch={arch}", "-std=c++17", "-O3",
                           "-DCUTLASS_ARCH_MMA_SM80_SUPPORTED"],
        verbose=False,
    )


def _prebuilt_so_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "lib", "cutlass_conv3d_fp16acc_ext.so")


def _load_op():
    """Register torch.ops.fp16acc.* — prefer prebuilt .so, else JIT compile.

    Idempotent: no-op if the namespace is already registered.
    """
    if hasattr(torch.ops, "fp16acc") and hasattr(torch.ops.fp16acc, "cutlass_conv3d"):
        return True
    so = _prebuilt_so_path()
    if os.path.exists(so):
        torch.ops.load_library(so)          # zero-cost, registers the TORCH_LIBRARY
    else:
        _build_extension()                  # JIT fallback (~90s first call)
    return True


_ext = None
_patched = False
_orig_forward = None


def _ensure_ext():
    """Register the CUTLASS op (once). Meta impl is in C++ (no register_fake)."""
    global _ext
    _load_op()
    _ext = True   # marker: op is loaded (kept for backward-compat with old callers)
    return _ext


def install(force: bool = False) -> bool:
    """Install the fp16-acc patch on Encoder3d.forward.

    Returns True if patched, False if skipped (unsupported arch, already patched,
    or disabled by flag).

    Args:
      force: bypass the sm_capability check (for debug on unsupported hardware).
    """
    global _patched, _orig_forward
    if _patched:
        return False
    if not force and not _is_supported_arch():
        # _is_supported_arch() already logged the specific reason
        return False
    _ensure_ext()
    # Import the streaming impl lazily so importers can pick it up post-hoc.
    from . import ndhwc_encode_e2e as e2e
    _orig_forward = Encoder3d.forward
    e2e.patch(True)
    _patched = True
    _log("NDHWC streaming Encoder3d installed; 2D Conv2d -> channels_last")
    return True


def uninstall() -> None:
    """Restore the original Encoder3d.forward."""
    global _patched, _orig_forward
    if not _patched or _orig_forward is None:
        return
    Encoder3d.forward = _orig_forward
    _patched = False
    _log("NDHWC patch uninstalled")


@contextmanager
def scoped_install():
    """Install the patch for a `with` block and uninstall on exit.

    Useful for restricting fp16-acc to a specific VAE.encode call while other
    code (e.g. EMA VAE, validation VAE) keeps the original path.

    Example:
        with vae_install.scoped_install():
            latent = vae.encode(x)
    """
    was_installed = _patched
    if not was_installed:
        install()
    try:
        yield
    finally:
        if not was_installed:
            uninstall()


# NOTE: no module-level auto-install here. Activation is driven by the
# wan_vae_fp16acc.pth site-hook -> wan_vae_fp16acc._boot.bootstrap() -> install().
# (Keeps importing this module side-effect-free for tooling/tests.)

"""wan_vae_fp16acc — fp16-acc CausalConv3d patch for Wan2.2 VAE encoder on consumer
GeForce cards (RTX 20/30/40/50). Gives 2x Tensor-Core throughput via fp16
accumulation while keeping end-to-end latent precision.

Activation (auto, via .pth site-hook): set env  COSMOS_VAE_FP16ACC=1
Activation (explicit):
    import wan_vae_fp16acc
    wan_vae_fp16acc.install()

On unsupported archs (data-center / workstation: A100/H100/B200/PRO 6000)
install() silently no-ops — fp16-acc = fp32-acc there, no gain.

NOTE: this module is imported by the .pth site-hook on EVERY interpreter start.
It must stay lightweight — no `torch` / `cosmos_framework` imports at module
level. Heavy imports happen lazily inside the functions (only when the env flag
is set and install() actually runs).
"""
__version__ = "0.1.0"
__all__ = ["install", "uninstall", "scoped_install", "is_supported_arch"]


def install(force: bool = False) -> bool:
    from .vae_install import install as _install
    return _install(force=force)


def uninstall() -> None:
    from .vae_install import uninstall as _uninstall
    _uninstall()


def scoped_install():
    from .vae_install import scoped_install as _scoped
    return _scoped()


def is_supported_arch() -> bool:
    from .vae_install import is_supported_arch as _is_supported
    return _is_supported()

"""Bootstrap entry point invoked by the site-hook (.pth file).

The .pth file (installed into site-packages root) runs:
    import wan_vae_fp16acc; wan_vae_fp16acc._boot.bootstrap()

bootstrap() checks COSMOS_VAE_FP16ACC and applies the patch at process start,
before the training framework imports anything. Failures are warnings only —
startup must never crash because of this hook.
"""
import os


def bootstrap() -> None:
    if os.environ.get("COSMOS_VAE_FP16ACC", "0") not in ("1", "true", "yes", "True"):
        return
    try:
        from . import vae_install
        vae_install.install()
    except Exception as e:  # noqa: BLE001 — never crash startup
        print(f"[wan_vae_fp16acc] WARN: bootstrap failed: {e!r}", flush=True)

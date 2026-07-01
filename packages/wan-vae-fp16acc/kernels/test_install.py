"""Quick test: explicit install() applies the patch + WanVAE.encode works.

NOTE: importing `vae_install` does NOT auto-patch (activation is driven by the
.pth site-hook -> bootstrap() -> install()). Tests must call install() explicitly.
"""
import os
import torch

import wan_vae_fp16acc
from wan_vae_fp16acc import vae_install
from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import WanVAE

# Explicit activation + verify the patch actually landed (guards against the
# old "import side-effect" assumption and against arch-gate silent skips).
assert wan_vae_fp16acc.install(), "install() returned False — arch gate skipped or already patched"
assert vae_install._patched, "Encoder3d.forward was not patched"

vae = WanVAE(
    z_dim=48,
    vae_pth=os.environ.get("WAN_VAE_PATH", "/pfs/pfs-iQ14no/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    dtype=torch.bfloat16,
    device="cuda",
    is_amp=False,
    temporal_window={"256": 68, "480": 24, "720": 12},
    encode_exact_durations=[17, 61, 73],
)
vae.model.eval()
x = torch.randn(1, 3, 17, 544, 736, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    out = vae.encode(x)
print(f"OK shape={tuple(out.shape)} nan={torch.isnan(out).any().item()} max_abs={out.abs().max().item():.3f}")

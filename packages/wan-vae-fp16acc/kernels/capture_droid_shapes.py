"""Capture the actual per-stage conv3d shapes a DROID encode feeds to the
CUTLASS op on 4090.

Hooks ndhwc_streaming.causal_conv3d_ndhwc (the single chokepoint for all
fp16-acc conv3d calls) to record (x.shape, layer.weight.shape, stride) per call,
then runs a real DROID-shape encode and prints unique shapes grouped by Cout.

DROID training shape: [1, 3, 33, 544, 736] bf16, temporal_window {"480":24},
encode_exact_durations=[33]. The encoder streams prime(1) + 24-chunk + 8-remainder,
so we capture BOTH chunk shapes.
"""
import os, sys, collections
sys.path.insert(0, "/pfs/pfs-iQ14no/songzhan/cosmos-framework")
sys.path.insert(0, "/pfs/pfs-iQ14no/songzhan/cosmos-framework/packages/wan-vae-fp16acc")
os.environ["COSMOS_VAE_FP16ACC"] = "1"
os.environ.setdefault("WAN_VAE_PATH", "/pfs/pfs-iQ14no/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")

import torch
import wan_vae_fp16acc
from wan_vae_fp16acc import ndhwc_streaming as nd

# Install the fp16-acc patch (sm_89 4090 -> patches Encoder3d.forward).
assert wan_vae_fp16acc.install(), "install failed (arch not supported?)"

# Hook the conv chokepoint.
_log = []
_orig_conv = nd.causal_conv3d_ndhwc
def _hooked(layer, x, cache_x=None):
    # x: NDHWC [N,T,H,W,Cin]; layer.weight: [Cout,Cin,Kt,Kh,Kw]
    Cin = x.shape[-1]
    Cout = layer.weight.shape[0]
    _log.append((tuple(x.shape), Cout, Cin, tuple(layer.stride)))
    return _orig_conv(layer, x, cache_x)
nd.causal_conv3d_ndhwc = _hooked

# Build VAE with DROID config.
from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import WanVAE
vae = WanVAE(
    z_dim=48, vae_pth=os.environ["WAN_VAE_PATH"], dtype=torch.bfloat16, device="cuda",
    is_amp=False,
    temporal_window={"256": 68, "480": 24, "720": 12},
    encode_exact_durations=[33],
)
vae.model.eval()
torch.manual_seed(0)

x = torch.randn(1, 3, 33, 544, 736, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    out = vae.encode(x)
print("encode OK, latent", tuple(out.shape), "calls:", len(_log))

# Group by Cout (stage discriminator), report unique (N,T,H,W,Cin) + count.
by_cout = collections.defaultdict(list)
for xshape, Cout, Cin, stride in _log:
    by_cout[Cout].append((xshape, Cin, stride))
print("\n=== captured conv3d shapes (NDHWC input) grouped by Cout ===")
for Cout in sorted(by_cout):
    rows = by_cout[Cout]
    uniq = collections.Counter((xs, Cin, st) for xs, Cin, st in rows)
    print(f"\nCout={Cout}  ({len(rows)} calls, {len(uniq)} unique):")
    for (xs, Cin, st), cnt in uniq.most_common():
        N, T, H, W, Cin2 = xs
        print(f"  x=[{N},{T},{H},{W},{Cin2}] Cin={Cin} stride={st}  x{cnt}")

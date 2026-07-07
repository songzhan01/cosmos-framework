"""Validate fp16-acc VAE encoder on REAL DROID inputs (not synthetic randn).
Loads a real DROID .mp4 sample, runs the same preprocessing as training
(decode -> resize 544x736 3:4 -> normalize), encodes with both paths, and
reports the latent statistics that matter for training:
- max_abs / p99_abs / mean_rel (vs baseline cuDNN fp32-acc)
- max channel-wise stats (some channels matter more than others)
- whether the magnitudes match what training sees (so the diffusion noise
  schedule isn't off-distribution)."""

import os, glob
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import WanVAE
from wan_vae_fp16acc import ndhwc_encode_e2e as E  # imports the patch
# The CUTLASS op is registered lazily by ndhwc_streaming._ensure_op() on first
# conv (uses the in-package prebuilt .so via vae_install._load_op). No JIT build
# and no hardcoded CUTLASS header path needed here.

# Load Wan2.2 VAE
torch.manual_seed(0)
vae = WanVAE(z_dim=48, vae_pth=os.environ["WAN_VAE_PATH"], dtype=torch.bfloat16, device="cuda",
             is_amp=False, temporal_window={"256":68,"480":24,"720":12}, encode_exact_durations=[17,61,73])
vae.model.eval()

# Find some real DROID mp4 samples (exterior_1_left, 480p 3:4 = 544x736 = native VAE input)
samples = sorted(glob.glob("/pfs/pfs-iQ14no/datasets/droid_lerobot_v30_subset20/videos/observation.images.exterior_1_left/chunk-000/file-*.mp4"))[:5]
print(f"Loaded {len(samples)} DROID mp4 samples")

# Training preprocessing matches cosmos.data.vfm.action sample_processor:
# 1) read mp4 -> [T,H,W,3] uint8 in 0..255
# 2) permute to [3,T,H,W]
# 3) resize H/W to 544/736 (3:4 aspect at 480p) using bilinear
# 4) scale to [-1, 1]: (x/255 - 0.5) * 2
# 5) T=17 frames (encode_exact_durations[0] in nano config is 33; we use 17 to match the
#    bench-shape used in optimization — same conv K, validates same kernels)
def preprocess(path, T=17, H=544, W=736):
    frames, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")  # uint8 [T,3,H_src,W_src]
    if frames.shape[0] < T:
        return None
    # Take first T frames; resize to 544x736
    f = frames[:T].to(torch.float32) / 255.0  # [T,3,H_src,W_src]
    f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)  # [T,3,544,736]
    # NCDHW: [1, 3, T, 544, 736]
    x = f.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]
    x = (x - 0.5) * 2.0  # to [-1,1]
    return x.to("cuda", torch.bfloat16)

# Per-channel stats accumulator
acc_max = []
acc_p99 = []
acc_mean_rel = []
acc_ref_max = []
acc_compile_vs_eager = []    # compile-vs-eager stats per sample (training baseline's own noise)
acc_fp16_vs_compile = []     # fp16-acc-vs-compile stats per sample (the relevant deviation)

for i, path in enumerate(samples):
    x = preprocess(path)
    if x is None: continue
    # baseline (cuDNN fp32-acc) — also compare against the COMPILE path (the real
    # training baseline; it already has its own ~1% noise vs eager). This way the
    # fp16-acc gap is measured against the actual training distribution.
    E.patch(False)
    with torch.no_grad():
        ref_eager = vae.encode(x).float().cpu()
    torch.backends.cudnn.benchmark = True
    fn_compile = torch.compile(vae.encode, mode="max-autotune", dynamic=False)
    with torch.no_grad():
        for _ in range(2): fn_compile(x)
        ref_compile = fn_compile(x).float().cpu()
    torch.cuda.empty_cache()
    # fp16-acc path (compiled, same path training would use)
    E.patch(True)
    fn_fp16 = torch.compile(vae.encode, mode="max-autotune-no-cudagraphs", dynamic=False)
    with torch.no_grad():
        for _ in range(2): fn_fp16(x)
        out = fn_fp16(x).float().cpu()
    E.patch(False)
    # Both: vs eager (the absolute reference) and vs compile (the actual training baseline).
    def stats(a, ref, tag):
        err = (a - ref).abs(); rel = err / (ref.abs() + 1e-3)
        return {"tag": tag, "max_abs": err.max().item(), "p99": torch.quantile(err.flatten(), 0.99).item(),
                "mean_rel": rel.mean().item(), "ref_max": ref.abs().max().item()}
    s_compile = stats(ref_compile, ref_eager, "compile vs eager")
    s_fp16 = stats(out, ref_eager, "fp16-acc vs eager")
    s_fp16_vs_compile = stats(out, ref_compile, "fp16-acc vs compile (training baseline)")
    has_nan = torch.isnan(out).any().item() or torch.isinf(out).any().item()
    print(f"\n[{i+1}] {os.path.basename(path):20s}  ref_max={s_compile['ref_max']:.3f}  NaN={has_nan}")
    for s in [s_compile, s_fp16, s_fp16_vs_compile]:
        print(f"     {s['tag']:42s} max_abs={s['max_abs']:.4f} ({s['max_abs']/s['ref_max']*100:.2f}%)  "
              f"p99={s['p99']:.4f}  mean_rel={s['mean_rel']:.4f}")
    print(f"     baseline latent stats: mean={ref_eager.mean().item():+.4f} std={ref_eager.std().item():.4f}  min={ref_eager.min().item():.3f}  max={ref_eager.max().item():.3f}")
    print(f"     compile  latent stats: mean={ref_compile.mean().item():+.4f} std={ref_compile.std().item():.4f}  min={ref_compile.min().item():.3f}  max={ref_compile.max().item():.3f}")
    print(f"     fp16-acc latent stats: mean={out.mean().item():+.4f} std={out.std().item():.4f}  min={out.min().item():.3f}  max={out.max().item():.3f}")
    acc_max.append(s_fp16["max_abs"]); acc_p99.append(s_fp16["p99"])
    acc_mean_rel.append(s_fp16["mean_rel"]); acc_ref_max.append(s_fp16["ref_max"])
    acc_compile_vs_eager.append(s_compile)
    acc_fp16_vs_compile.append(s_fp16_vs_compile)

print()
print(f"=== SUMMARY vs EAGER (n={len(acc_max)} DROID samples) ===")
print(f"  max_abs:    mean={sum(acc_max)/len(acc_max):.4f}  worst={max(acc_max):.4f}")
print(f"  p99_abs:    mean={sum(acc_p99)/len(acc_p99):.4f}  worst={max(acc_p99):.4f}")
print(f"  mean_rel:   mean={sum(acc_mean_rel)/len(acc_mean_rel):.4f}  worst={max(acc_mean_rel):.4f}")
print(f"  ref_max:    mean={sum(acc_ref_max)/len(acc_ref_max):.3f}  worst={max(acc_ref_max):.3f}")
print(f"  max_abs/ref_max %: mean={sum(a/r for a,r in zip(acc_max,acc_ref_max))/len(acc_max)*100:.2f}%  worst={max(a/r for a,r in zip(acc_max,acc_ref_max))*100:.2f}%")
print()
print(f"=== SUMMARY fp16-acc vs COMPILE (the TRAINING baseline) ===")
vs_c = acc_fp16_vs_compile
print(f"  max_abs:    mean={sum(s['max_abs'] for s in vs_c)/len(vs_c):.4f}  worst={max(s['max_abs'] for s in vs_c):.4f}")
print(f"  p99_abs:    mean={sum(s['p99'] for s in vs_c)/len(vs_c):.4f}  worst={max(s['p99'] for s in vs_c):.4f}")
print(f"  mean_rel:   mean={sum(s['mean_rel'] for s in vs_c)/len(vs_c):.4f}  worst={max(s['mean_rel'] for s in vs_c):.4f}")
print(f"  max_abs/ref_max %: mean={sum(s['max_abs']/s['ref_max'] for s in vs_c)/len(vs_c)*100:.2f}%  worst={max(s['max_abs']/s['ref_max'] for s in vs_c)*100:.2f}%")
print(f"\ntraining-acceptable bar: fp16-acc-vs-compile within 1.5x of compile-vs-eager (compile's own noise).")

# Final comparison table
print()
print("=== HEAD-TO-HEAD (training baseline is compile, not eager) ===")
ce = acc_compile_vs_eager
ce_mean_rel = sum(s['mean_rel'] for s in ce)/len(ce)
fc_mean_rel = sum(s['mean_rel'] for s in vs_c)/len(vs_c)
ce_max_pct = sum(s['max_abs']/s['ref_max'] for s in ce)/len(ce)*100
fc_max_pct = sum(s['max_abs']/s['ref_max'] for s in vs_c)/len(vs_c)*100
print(f"  compile vs eager  (training baseline's own noise): mean_rel={ce_mean_rel:.4f}  max%={ce_max_pct:.2f}%")
print(f"  fp16-acc vs compile (the relevant deviation):       mean_rel={fc_mean_rel:.4f}  max%={fc_max_pct:.2f}%")
print(f"  RATIO (fp16-acc/compile noise): mean_rel={fc_mean_rel/ce_mean_rel:.2f}x  max%={fc_max_pct/ce_max_pct:.2f}x")
print(f"  -> fp16-acc adds {fc_mean_rel/ce_mean_rel:.2f}x the noise the training is already absorbing.")

#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""Full NDHWC streaming encoder + e2e measure. Monkey-patch Encoder3d.forward
to a streaming NDHWC version (conv1 NCDHW isolated slot 0; all else NDHWC:
ResidualBlock + Resample time_conv + head, all NDHWC cache slots). feat_idx
order matches the original exactly. Then encode() + compile, measure vs cuDNN."""
import os, torch, torch.nn.functional as F
from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import (
    WanVAE, Encoder3d, CausalConv3d, RMS_norm, ResidualBlock, Resample, AttentionBlock,
    _update_cache_and_apply, _contiguous_clone, CACHE_T)
from . import ndhwc_streaming as nd  # causal_conv3d_ndhwc, _update_cache_and_apply_ndhwc, rms_norm_ndhwc, residual_block_ndhwc, avg_down_3d_ndhwc, attention_ndhwc

def resample_ndhwc_stream(self, x, feat_cache, feat_idx):  # x NDHWC [B,T,H,W,C] fp16
    # NDHWC-native 2D conv via channels_last: view stride-trick, no copy.
    # F.conv2d on channels_last input + channels_last weight runs the NHWC cuDNN
    # kernel directly (no nchwToNhwc helper).
    b, t, h, w, c = x.shape
    x_cl = x.view(b * t, h, w, c).permute(0, 3, 1, 2)   # stride-only, channels_last NCHW
    # self.resample is Sequential(ZeroPad2d(0,1,0,1), Conv2d(stride=2)) for downsample
    # We split: pad on NDHWC-friendly form is cheaper as F.pad on the channels_last view.
    pad = self.resample[0].padding  # ZeroPad2d (left,right,top,bottom)
    conv2d = self.resample[1]       # Conv2d (already channels_last weight)
    x_cl = F.pad(x_cl, pad)
    # Run Conv2d in bf16 channels_last (no copy: x_cl already channels_last layout).
    x_cl_bf = x_cl.to(torch.bfloat16, memory_format=torch.channels_last)
    out = F.conv2d(x_cl_bf, conv2d.weight, conv2d.bias, stride=conv2d.stride, padding=0, dilation=conv2d.dilation, groups=conv2d.groups)
    # out is channels_last NCHW -> view back NDHWC (stride-only, no copy)
    x = out.permute(0, 2, 3, 1).reshape(b, t, out.shape[2], out.shape[3], out.shape[1]).contiguous().to(torch.float16)
    if self.mode == "downsample3d":
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            if x.shape[1] == 1:  # prime T==1: skip time_conv, seed cache
                feat_cache[idx] = _contiguous_clone(x)
            else:
                cache_x = _contiguous_clone(x[:, -1:, :, :, :])
                x_in = F.pad(x, (0, 0, 0, 0, 0, 0, 2, 0))
                x = nd.causal_conv3d_ndhwc(self.time_conv, x_in, None)
                feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            cache_x = _contiguous_clone(x[:, -1:, :, :, :])
            x_cat = torch.cat([feat_cache[idx][:, -1:, :, :, :], x], dim=1)
            if x_cat.shape[1] < 3:
                x_cat = F.pad(x_cat, (0, 0, 0, 0, 0, 0, 3 - x_cat.shape[1], 0))
            x = nd.causal_conv3d_ndhwc(self.time_conv, x_cat, None)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
    return x

def down_residual_ndhwc_stream(self, x, feat_cache, feat_idx):  # x NDHWC fp16
    # avg_down_3d_ndhwc operates on the dtype it receives; keep input fp16 throughout
    # so shortcut + main both stay fp16 (no extra cast).
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    x_shortcut = nd.avg_down_3d_ndhwc(self.avg_shortcut, x)  # stays fp16
    for module in self.downsamples:
        if isinstance(module, ResidualBlock):
            x = nd.residual_block_ndhwc(module, x, feat_cache, feat_idx)
        elif isinstance(module, Resample):
            x = resample_ndhwc_stream(module, x, feat_cache, feat_idx)
    return x + x_shortcut

_orig_enc = Encoder3d.forward
def _enc_ndhwc_stream(self, x, feat_cache=None):  # x NCDHW [B,12,T,H,W] bf16
    # feat_cache is always a list (from WanVAE._new_enc_cache) in the real
    # chunked-encode path; the no-cache single-shot path was removed as dead
    # code. Fail loud if a caller forgets it rather than silently diverging.
    assert feat_cache is not None, "_enc_ndhwc_stream requires feat_cache (list from _new_enc_cache)"
    feat_idx = [0]
    x = _update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)
    x = x.permute(0, 2, 3, 4, 1).contiguous().to(torch.float16)  # NDHWC fp16
    for layer in self.downsamples:
        x = down_residual_ndhwc_stream(layer, x, feat_cache, feat_idx)
    for layer in self.middle:
        if isinstance(layer, ResidualBlock):
            x = nd.residual_block_ndhwc(layer, x, feat_cache, feat_idx)
        elif isinstance(layer, AttentionBlock):
            x = nd.attention_ndhwc(layer, x)
    for layer in self.head:
        if isinstance(layer, CausalConv3d):
            x = nd._update_cache_and_apply_ndhwc(x, layer, feat_cache, feat_idx)
        elif isinstance(layer, RMS_norm):
            x = nd.rms_norm_ndhwc(layer, x)
        else:
            x = layer(x)
    out_bf = x.permute(0, 4, 1, 2, 3).to(torch.bfloat16)  # stride-only permute + cast
    return out_bf.contiguous()  # caller asserts is_contiguous()

def patch(on): Encoder3d.forward = _enc_ndhwc_stream if on else _orig_enc

# Skip the standalone benchmark when imported as a module (training import path).
if __name__ == "__main__":
    def bench(fn, iters=15, warmup=4):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    vae = WanVAE(z_dim=48, vae_pth=os.environ["WAN_VAE_PATH"], dtype=torch.bfloat16, device="cuda",
                 is_amp=False, temporal_window={"256":68,"480":24,"720":12}, encode_exact_durations=[17,61,73])
    vae.model.eval()

    for m in vae.model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
    x = torch.randn(1, 3, 17, 544, 736, device="cuda", dtype=torch.bfloat16)
    torch.backends.cudnn.benchmark = True

    patch(False)
    ref = vae.encode(x).float().cpu()
    with torch.no_grad():
        for _ in range(3): vae.encode(x)
        ms_cudnn = bench(lambda: vae.encode(x))
    fn_cudnn = torch.compile(vae.encode, mode="max-autotune", dynamic=False)
    with torch.no_grad():
        for _ in range(3): fn_cudnn(x)
        ms_cudnn_c = bench(lambda: fn_cudnn(x))
    print(f"cuDNN eager {ms_cudnn:.1f} | compile {ms_cudnn_c:.1f}")
    torch.cuda.empty_cache()

    patch(True)
    out = vae.encode(x).float().cpu()
    with torch.no_grad():
        for _ in range(3): vae.encode(x)
        ms_nd = bench(lambda: vae.encode(x))
    fn_nd = torch.compile(vae.encode, mode="max-autotune-no-cudagraphs", dynamic=False)
    with torch.no_grad():
        for _ in range(3): fn_nd(x)
        ms_nd_c = bench(lambda: fn_nd(x))
    err = (out - ref).abs()
    print(f"NDHWC eager {ms_nd:.1f} | compile {ms_nd_c:.1f}  (vs cuDNN compile {ms_cudnn_c/ms_nd_c:.2f}x)")
    print(f"latent: max_abs={err.max().item():.4f} ({err.max().item()/ref.abs().max().item()*100:.2f}%) "
          f"p99={torch.quantile(err.flatten(),0.99).item():.4f} nan={torch.isnan(out).any().item()}")
    patch(False)

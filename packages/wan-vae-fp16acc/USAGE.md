# wan-vae-fp16acc 使用指南

Wan2.2 VAE encoder 的 fp16-acc CausalConv3d 补丁包。在消费级 GeForce（RTX 20/30/40/50）上，把 VAE encoder 的 conv3d 从 cuDNN fp32-acc 换成 CUTLASS fp16-acc（`ElementAccumulator = half_t` → `mma.sync.f16.f16.f16.f16`），利用消费卡 fp16-acc 比 fp32-acc 快 2× 的 Tensor-Core 吞吐，端到端 encoder 提速 **~1.45×（SM89 / RTX 4090）**，精度可控。

> 数据中心 / 工作站卡（A100/H100/B200/RTX PRO 6000 等）fp16-acc = fp32-acc，无 uplift，补丁自动跳过（`is_supported_arch()` 门控）。

---

## 1. 架构与文件分层

| 层 | 文件 | 作用 |
|---|---|---|
| 打包激活 | `pyproject.toml`, `setup.py`, `wan_vae_fp16acc.pth`, `__init__.py`, `_boot.py` | pip 装包 + `.pth` 启动钩子 + env 门控自动激活 |
| 运行时编排 | `vae_install.py` | `install()/uninstall()/scoped_install()`、arch 门控、`_load_op()`（预编译 .so 优先，JIT 回退）、monkey-patch `Encoder3d.forward` |
| NDHWC 流式 encoder | `ndhwc_streaming.py`, `ndhwc_encode_e2e.py` | conv3d 原语 + `Encoder3d.forward` 的 NDHWC 流式替换（时序分块降显存） |
| CUTLASS kernel | `cutlass_conv3d_ext.cu`, `build_so.sh` | fp16-acc Conv3d kernel + `torch.ops.fp16acc.*` 注册 + 预编译 .so 构建 |
| 调优/测试 | `kernels/` | tile 调优、形状捕获、精度/性能验证 |

---

## 2. 构建（预编译 .so）

在 4090（或有 nvcc + CUTLASS 的机器）上跑：

```bash
cd packages/wan-vae-fp16acc
bash build_so.sh
```

`build_so.sh` 做的事：
- **CUTLASS pin 到固定 commit** `b9847690`（可复现）；无 `CUTLASS_PATH` 时自动 clone 到缓存
- 委托 `torch.utils.cpp_extension.load()` 编译（自动解析 torch 头/链接库）
- arch 自动检测（4090 → `sm_89`）
- `TMPDIR` 重定向（nvcc 编 CUTLASS 模板会产生 GB 级中间文件，/tmp 常满）
- 产物：`wan_vae_fp16acc/lib/cutlass_conv3d_fp16acc_ext.so`（~830 KB）

网络不通时设代理：`export https_proxy=http://192.168.112.80:18000`

---

## 3. 安装与激活

```bash
pip install .          # 装包 + .pth 落到 site-packages 根
```

三种激活方式：

```bash
# A. 自动（推荐）：env flag，.pth → bootstrap() → install()
export COSMOS_VAE_FP16ACC=1
python train.py

# B. 显式
import wan_vae_fp16acc
wan_vae_fp16acc.install()

# C. 临时（with 块内激活，退出还原）
with wan_vae_fp16acc.scoped_install():
    latent = vae.encode(x)
```

不设 env 则零侵入（bootstrap no-op）。

**冒烟验证**：
```bash
python kernels/test_install.py    # install() 生效 + encode 不崩 + 无 NaN
```

---

## 4. Tile 调优工作流

CUTLASS threadblock tile（128×128 / 128×256 / 256×128 / 64×256）按 conv 的 Cout 分阶段选择，`select_tile(K)` 默认值（K=160→128×256，K=320/640→256×128）已对 DROID 形状在 4090 上验证最优。**输入形状变了（换数据集/分辨率/T）需重跑调优。**

### 4.1 捕获真实形状

```bash
python kernels/capture_droid_shapes.py
```

hook `causal_conv3d_ndhwc`（所有 fp16-acc conv3d 的唯一入口），跑真实 DROID encode `[1,3,33,544,736]`，按 Cout 分组打印主导形状。DROID 当前形状（4090 实测）：

| stage | Cout | 主导形状 (N,Cin,T,H,W,Cout) |
|---|---|---|
| 0 | 160 | (1,160,24,272,368,160) |
| 1 | 320 | (1,320,24,136,184,320) |
| 2 | 640 | (1,640,12,68,92,640) |

### 4.2 扫最优 tile

```bash
# 快速（1×50 iters）
python kernels/sweep_tiles.py

# 严格（3×200 iters 取 median，结果接近噪声 floor 时用）
python kernels/sweep_tiles.py --rounds 3 --iters 200
```

输出每 stage 4 种 tile 的 ms/TFLOPS，标 `(default)` 和 `[keep/CHANGE]` 判定。

> **噪声注意**：快速模式在 ~2% 差距时可能误判（如 stage2 快速模式可能显示 `[CHANGE]`）。结果接近噪声 floor 时务必用 `--rounds 3` 复测。

### 4.3 回填 + 重建（仅当某个 stage 显示 `[CHANGE]`）

1. 编辑 `wan_vae_fp16acc/cutlass_conv3d_ext.cu::select_tile()`，把该 stage 的默认值改成最优 tile 编号
2. `bash build_so.sh` 重编 .so
3. `python kernels/test_cutlass_correctness.py` 验证

**当前 DROID 形状下三个 stage 都是 `[keep]`，无需改动。**

### 4.4 运行时动态切 tile（不重编）

```python
import torch
# 临时覆盖某 stage 的 tile（>=0 覆盖, -1 用默认）
torch.ops.fp16acc.set_tile_overrides(s0, s1, s2)
```

进程级、易失，用于不改 .so 快速验证某 tile 配置。

---

## 5. 精度验证

fp16-acc 把累加器从 fp32 降到 fp16，必须验证 latent 精度不影响训练。

### 5.1 kernel 级（vs cuDNN fp32-acc）

```bash
python kernels/test_cutlass_correctness.py
```

对比 CUTLASS fp16-acc conv3d 与 `F.conv3d`（fp32-acc cuDNN）的 max_abs，并报 TFLOPS。改了 `.cu` / 重建 .so 后必跑。

### 5.2 端到端 proxy（im2col + Triton fp16-acc matmul）

```bash
python kernels/validate_e2e_precision.py
```

monkey-patch 每个 CausalConv3d 走 im2col + naive fp16-acc matmul，跑完整 encode，对比 latent vs fp32-acc encoder 的 max_abs / mean_rel。验证 fp16-acc **算术**精度安全（mma 累加器 dtype 与 CUTLASS 一致，结论可迁移）。速度不代表真实 kernel。

### 5.3 真实 DROID 数据（上线前必跑）

```bash
DROID_MP4=<path> WAN_VAE_PATH=... python kernels/validate_real_droid.py
```

加载真实 DROID .mp4 → 训练同款预处理（decode→544×736 3:4→normalize）→ 双路径 encode → 报训练关心的统计：
- `max_abs / p99_abs / mean_rel`（vs baseline cuDNN fp32-acc）
- 通道级统计
- 幅度是否在训练分布内（避免 diffusion noise schedule 偏分布）

**精度判据**：VAE compile 路径本身有 ~1% 噪声，fp16-acc 的 latent max_abs 应在该量级内（参考 4090 实测：max_abs ~0.05 量级，相对 <1%）。

---

## 6. 性能验证

### 6.1 kernel 级 TFLOPS

`test_cutlass_correctness.py` 每 stage 报 TFLOPS。4090 + DROID 形状实测（3×200 iters）：

| stage | tile | TFLOPS |
|---|---|---|
| 0 (Cout=160) | 128×256 | 183 T |
| 1 (Cout=320) | 256×128 | 221 T |
| 2 (Cout=640) | 256×128 | 246 T |

### 6.2 端到端 encoder 提速

`ndhwc_encode_e2e.py` 的 `__main__` 块对比 cuDNN vs NDHWC fp16-acc（eager + compile）：

```bash
WAN_VAE_PATH=... python -m wan_vae_fp16acc.ndhwc_encode_e2e
```

输出 `cuDNN eager/compile` vs `NDHWC eager/compile` 的 ms 和加速比，以及 latent max_abs。SM89 上端到端 ~1.45×。

### 6.3 tile sweep 性能

`kernels/sweep_tiles.py` 输出每 tile 的 ms + TFLOPS，用于判断 tile 选择是否最优（见 §4）。

---

## 7. 约束与注意

- **torch.compile**：用 `mode="max-autotune-no-cudagraphs"`。cudagraphs 与 op 的 per-shape kernel cache 冲突。
- **VAE 只读**：仅支持 eval（推理/预编码）。训练中 live-update VAE 不支持。
- **cudnn.benchmark=True**：训练 config 默认开，保持。
- **arch 门控**：消费 GeForce（sm_75/86/89/120-GeForce）激活；数据中心/工作站（A100/H100/B200/PRO 6000）自动跳过。`force=True` 可强制（调试用）。
- **A100 特例**：有 2× uplift 但属数据中心，默认拒绝；`FP16ACC_ALLOW_A100=1` 可 override（未测试）。

---

## 8. 故障排查

| 现象 | 原因 / 修复 |
|---|---|
| `git clone CUTLASS failed` | 网络，设 `https_proxy=http://192.168.112.80:18000` 或 `CUTLASS_PATH` 指本地 checkout |
| nvcc `No space left on device` | /tmp 满，`build_so.sh` 已重定向 `TMPDIR`，可 `export TMPDIR=...` 自定义 |
| `set_tile_overrides` 调用崩 | 旧 .so 的注册 bug（无 tensor 参数 op 只注册 CUDA impl）。已修复，重编 .so 即可 |
| BS=32 OOM | 4090 24GB 上限 BS≈4（DROID T=33）；大 BS 用 B200/PRO 6000（但补丁在那些卡上跳过） |
| compile 报 cudagraphs 冲突 | 用 `max-autotune-no-cudagraphs`，不要 `reduce-overhead` |

---

## 9. 典型工作流速查

**首次部署**：
```bash
bash build_so.sh              # 4090 上编 .so
pip install .                 # 装包
python kernels/test_install.py
python kernels/test_cutlass_correctness.py
python kernels/validate_real_droid.py
export COSMOS_VAE_FP16ACC=1   # 训练时激活
```

**换数据集 / 形状变了**：
```bash
python kernels/capture_droid_shapes.py        # 1. 采新形状
# 编辑 sweep_tiles.py shapes
python kernels/sweep_tiles.py --rounds 3      # 2. 扫最优 tile
# 若 [CHANGE]: 回填 select_tile → build_so.sh → test_cutlass_correctness.py
python kernels/validate_real_droid.py         # 3. 重验精度
```

**改了 kernel**：
```bash
bash build_so.sh
python kernels/test_cutlass_correctness.py    # kernel 正确性 + TFLOPS
python kernels/test_install.py                # 补丁生效
python kernels/validate_real_droid.py         # 真实数据精度
```

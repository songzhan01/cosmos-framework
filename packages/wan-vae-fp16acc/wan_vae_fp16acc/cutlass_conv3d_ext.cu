// SPDX-License-Identifier: BSD-3-Clause
// torch extension: CUTLASS fp16-acc Conv3d fprop callable from Python.
// A: [N,D,H,W,C] fp16 (NDHWC), B: [K,T,R,S,C] fp16 (KTRSC), returns [N,D,H,W,K] fp16.
// pad=1, stride=1, k=3 assumed (Z=D,P=H,Q=W). ElementAccumulator=half_t (2x TC).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute     = cutlass::half_t;

// Tilings sweep: per-stage optimal tile differs.
// Stage0 (M=160 small, N=spatial huge) -> needs wide-N tile (128x256).
// Stage1/2 (M=320/640, smaller N) -> standard 128x128.
template<int BM, int BN, int BK, int Stages> struct ConvCfg;
template<int Stages> struct ConvCfg<128, 128, 32, Stages> {
    using Kernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
        ElementA, cutlass::layout::TensorNDHWC,
        ElementB, cutlass::layout::TensorNDHWC,
        ElementC, cutlass::layout::TensorNDHWC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd>::Kernel;
};
template<int Stages> struct ConvCfg<128, 256, 32, Stages> {
    using Kernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
        ElementA, cutlass::layout::TensorNDHWC,
        ElementB, cutlass::layout::TensorNDHWC,
        ElementC, cutlass::layout::TensorNDHWC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd>::Kernel;
};
template<int Stages> struct ConvCfg<256, 128, 32, Stages> {
    using Kernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
        ElementA, cutlass::layout::TensorNDHWC,
        ElementB, cutlass::layout::TensorNDHWC,
        ElementC, cutlass::layout::TensorNDHWC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd>::Kernel;
};
template<int Stages> struct ConvCfg<64, 256, 32, Stages> {
    using Kernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
        ElementA, cutlass::layout::TensorNDHWC,
        ElementB, cutlass::layout::TensorNDHWC,
        ElementC, cutlass::layout::TensorNDHWC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd>::Kernel;
};

using Conv3dFprop_128x128 = cutlass::conv::device::ImplicitGemmConvolution<ConvCfg<128,128,32,3>::Kernel>;
using Conv3dFprop_128x256 = cutlass::conv::device::ImplicitGemmConvolution<ConvCfg<128,256,32,3>::Kernel>;
using Conv3dFprop_256x128 = cutlass::conv::device::ImplicitGemmConvolution<ConvCfg<256,128,32,3>::Kernel>;
using Conv3dFprop_64x256  = cutlass::conv::device::ImplicitGemmConvolution<ConvCfg<64,256,32,3>::Kernel>;
using Layout = cutlass::layout::TensorNDHWC;

// Runtime tile overrides (Python-settable) for per-stage tile sweeps without recompile.
static int g_tile_s0 = -1;
static int g_tile_s1 = -1;
static int g_tile_s2 = -1;
static int select_tile(int K) {
    if (K == 160) return (g_tile_s0 >= 0) ? g_tile_s0 : 1;  // 128x256 best (185T)
    if (K == 320) return (g_tile_s1 >= 0) ? g_tile_s1 : 2;  // 256x128 best (228T)
    if (K == 640) return (g_tile_s2 >= 0) ? g_tile_s2 : 2;  // 256x128 best (269T)
    return 0;
}

template<typename Op>
static cutlass::Status run_one(typename Op::Arguments const& args, void* ws) {
    Op op;
    auto st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return st;
    st = op.initialize(args, ws);
    if (st != cutlass::Status::kSuccess) return st;
    return op();
}

// =============================================================================
// Workspace allocator: cudaMalloc for fastest fixed-address behavior (CUTLASS
// expects a persistent, non-fragmenting scratch pad). Using torch's
// CUDACachingAllocator here caused a ~25% perf regression, likely because
// DataPtr deleter callbacks and pool eviction interfered with the conv kernel.
// Cleanup is handled by a static destructor that runs at process exit.
// =============================================================================
struct WorkspaceCache {
    std::unordered_map<std::string, void*> map;
    ~WorkspaceCache() {
        for (auto& kv : map) {
            if (kv.second) cudaFree(kv.second);
        }
    }
};
static WorkspaceCache g_ws;

static void* get_or_alloc_workspace(std::string const& key, size_t bytes) {
    auto it = g_ws.map.find(key);
    if (it != g_ws.map.end()) return it->second;
    if (bytes == 0) {
        g_ws.map[key] = nullptr;
        return nullptr;
    }
    void* ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    g_ws.map[key] = ptr;
    return ptr;
}
// =============================================================================

// Forward declaration: the padded variant (defined below) is the SINGLE
// CUTLASS dispatch implementation — it handles both valid conv (pad=0) and
// folded-pad conv. The valid-conv entry points delegate to it with zero pad,
// so there is no duplicated dispatch / cache code.
torch::Tensor cutlass_conv3d_fp16acc_padded(torch::Tensor A, torch::Tensor B,
                                            int64_t sld, int64_t slh, int64_t slw,
                                            int64_t pad_t, int64_t pad_h, int64_t pad_w);

torch::Tensor cutlass_conv3d_fp16acc(torch::Tensor A, torch::Tensor B, int64_t sld, int64_t slh, int64_t slw) {
    // Valid conv (caller pre-padded, pad=0): delegate to the padded impl.
    return cutlass_conv3d_fp16acc_padded(A, B, sld, slh, slw, 0, 0, 0);
}

// =============================================================================
// CUTLASS variant that ABSORBS the padding (no F.pad).
// Conv3dProblemSize accepts pad_d/h/w; CUTLASS implicit-gemm uses an
// AnalyticIterator that handles out-of-bounds reads as zero, so we can pass the
// UNPADDED input and let CUTLASS read the implicit zeros. Big win on stage0/1
// where F.pad is heavy (1.8ms + 0.5ms per chunk = ~5ms across 4 res blocks * 2
// convs * 2 chunks at stage0).
// =============================================================================

// IteratorOptimized only handles pad=0. Use IteratorAnalytic when pad>0.
// DefaultConv3dFprop selects this automatically based on Conv3dProblemSize.
// So we can just pass pad_d=1,pad_h=1,pad_w=1 and skip F.pad on caller side.
//
// For temporal CAUSAL pad (T pad on LEFT only, not symmetric), CUTLASS standard
// fprop expects symmetric pad. So we cannot fold causal-T into CUTLASS; only the
// symmetric H/W pads are foldable. We expose a "fold_spatial_pad" flag.
torch::Tensor cutlass_conv3d_fp16acc_padded(torch::Tensor A, torch::Tensor B,
                                            int64_t sld, int64_t slh, int64_t slw,
                                            int64_t pad_t, int64_t pad_h, int64_t pad_w) {
    TORCH_CHECK(A.scalar_type() == torch::kFloat16 && B.scalar_type() == torch::kFloat16, "fp16 only");
    int N = A.size(0), D = A.size(1), H = A.size(2), W = A.size(3), C = A.size(4);
    int K = B.size(0), T = B.size(1), R = B.size(2), S = B.size(3);
    int Z = (D + 2*(int)pad_t - T) / (int)sld + 1;
    int P = (H + 2*(int)pad_h - R) / (int)slh + 1;
    int Q = (W + 2*(int)pad_w - S) / (int)slw + 1;
    auto out = torch::empty({N, Z, P, Q, K}, A.options());
    cutlass::conv::Conv3dProblemSize problem_size(
        N, D, H, W, C, K, T, R, S, Z, P, Q,
        (int)pad_t, (int)pad_h, (int)pad_w,
        (int)sld, (int)slh, (int)slw,
        1, 1, 1,
        cutlass::conv::Mode::kCrossCorrelation);
    Layout lA = Layout::packed(cutlass::make_Coord(N, D, H, W, C));
    Layout lB = Layout::packed(cutlass::make_Coord(K, T, R, S, C));
    Layout lD = Layout::packed(cutlass::make_Coord(N, Z, P, Q, K));
    cutlass::TensorRef<ElementC, Layout> ref_C(reinterpret_cast<ElementC*>(out.data_ptr<at::Half>()), lD);
    cutlass::TensorRef<ElementC, Layout> ref_D(reinterpret_cast<ElementC*>(out.data_ptr<at::Half>()), lD);
    typename Conv3dFprop_128x128::Arguments args_128x128(problem_size,
        cutlass::TensorRef<ElementA, Layout>(reinterpret_cast<ElementA*>(A.data_ptr<at::Half>()), lA),
        cutlass::TensorRef<ElementB, Layout>(reinterpret_cast<ElementB*>(B.data_ptr<at::Half>()), lB),
        ref_C, ref_D, {ElementCompute(1), ElementCompute(0)},
        cutlass::conv::SplitKMode::kSerial);
    typename Conv3dFprop_128x256::Arguments args_128x256(problem_size,
        cutlass::TensorRef<ElementA, Layout>(reinterpret_cast<ElementA*>(A.data_ptr<at::Half>()), lA),
        cutlass::TensorRef<ElementB, Layout>(reinterpret_cast<ElementB*>(B.data_ptr<at::Half>()), lB),
        ref_C, ref_D, {ElementCompute(1), ElementCompute(0)},
        cutlass::conv::SplitKMode::kSerial);
    typename Conv3dFprop_256x128::Arguments args_256x128(problem_size,
        cutlass::TensorRef<ElementA, Layout>(reinterpret_cast<ElementA*>(A.data_ptr<at::Half>()), lA),
        cutlass::TensorRef<ElementB, Layout>(reinterpret_cast<ElementB*>(B.data_ptr<at::Half>()), lB),
        ref_C, ref_D, {ElementCompute(1), ElementCompute(0)},
        cutlass::conv::SplitKMode::kSerial);
    typename Conv3dFprop_64x256::Arguments args_64x256(problem_size,
        cutlass::TensorRef<ElementA, Layout>(reinterpret_cast<ElementA*>(A.data_ptr<at::Half>()), lA),
        cutlass::TensorRef<ElementB, Layout>(reinterpret_cast<ElementB*>(B.data_ptr<at::Half>()), lB),
        ref_C, ref_D, {ElementCompute(1), ElementCompute(0)},
        cutlass::conv::SplitKMode::kSerial);
    int tile = select_tile(K);
    static std::unordered_map<std::string, std::unique_ptr<Conv3dFprop_128x128>> cache_pa;
    static std::unordered_map<std::string, std::unique_ptr<Conv3dFprop_128x256>> cache_pb;
    static std::unordered_map<std::string, std::unique_ptr<Conv3dFprop_256x128>> cache_pc;
    static std::unordered_map<std::string, std::unique_ptr<Conv3dFprop_64x256>>  cache_pd;
    std::string key = std::to_string(reinterpret_cast<uintptr_t>(B.data_ptr<at::Half>()))
        + "_p_" + std::to_string(N) + "_" + std::to_string(D) + "_" + std::to_string(H) + "_" + std::to_string(W)
        + "_" + std::to_string(C) + "_" + std::to_string(K) + "_" + std::to_string(T) + "_" + std::to_string(R) + "_" + std::to_string(S)
        + "_" + std::to_string(Z) + "_" + std::to_string(P) + "_" + std::to_string(Q)
        + "_" + std::to_string((int)sld) + "_" + std::to_string((int)slh) + "_" + std::to_string((int)slw)
        + "_pad" + std::to_string(pad_t) + "_" + std::to_string(pad_h) + "_" + std::to_string(pad_w)
        + "_t" + std::to_string(tile);
    #define DISPATCH_P(T_, args, cache_v) {                                              \
        auto it = cache_v.find(key);                                                     \
        if (it == cache_v.end()) {                                                       \
            auto op = std::make_unique<T_>();                                            \
            if (op->can_implement(args) != cutlass::Status::kSuccess) { tile=0; goto FALLBACK_P; } \
            size_t ws = T_::get_workspace_size(args);                                    \
            /* torch caching allocator: no cudaMalloc leak */                            \
            void* wptr = get_or_alloc_workspace(key, ws);                                \
            TORCH_CHECK(op->initialize(args, wptr) == cutlass::Status::kSuccess, "init_p"); \
            TORCH_CHECK(op->operator()() == cutlass::Status::kSuccess, "run_p");         \
            cache_v[key] = std::move(op);                                                \
        } else {                                                                         \
            auto ws_it = g_ws.map.find(key);                                           \
            void* wptr = (ws_it != g_ws.map.end()) ? ws_it->second : nullptr;    \
            TORCH_CHECK(it->second->update(args, wptr) == cutlass::Status::kSuccess, "upd_p"); \
            TORCH_CHECK(it->second->operator()() == cutlass::Status::kSuccess, "runb_p");\
        }                                                                                \
    }
    if (tile == 1) { DISPATCH_P(Conv3dFprop_128x256, args_128x256, cache_pb); return out; }
    if (tile == 2) { DISPATCH_P(Conv3dFprop_256x128, args_256x128, cache_pc); return out; }
    if (tile == 3) { DISPATCH_P(Conv3dFprop_64x256,  args_64x256,  cache_pd); return out; }
FALLBACK_P:
    DISPATCH_P(Conv3dFprop_128x128, args_128x128, cache_pa);
    #undef DISPATCH_P
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_conv3d_fp16acc", &cutlass_conv3d_fp16acc, "CUTLASS fp16-acc Conv3d fprop (valid, stride)",
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("stride_d") = 1,
          pybind11::arg("stride_h") = 1, pybind11::arg("stride_w") = 1);
    m.def("cutlass_conv3d_fp16acc_padded", &cutlass_conv3d_fp16acc_padded,
          "CUTLASS fp16-acc Conv3d fprop with internal pad (no F.pad copy needed)",
          pybind11::arg("A"), pybind11::arg("B"),
          pybind11::arg("stride_d"), pybind11::arg("stride_h"), pybind11::arg("stride_w"),
          pybind11::arg("pad_t"), pybind11::arg("pad_h"), pybind11::arg("pad_w"));
    m.def("set_tile_overrides", [](int s0, int s1, int s2) {
        g_tile_s0 = s0; g_tile_s1 = s1; g_tile_s2 = s2;
    }, "Set per-stage tile (0=128x128, 1=128x256, 2=256x128, 3=64x256; -1=default).");
}

TORCH_LIBRARY(fp16acc, m) {
    m.def("cutlass_conv3d(Tensor A, Tensor B, int stride_d, int stride_h, int stride_w) -> Tensor");
    m.def("cutlass_conv3d_padded(Tensor A, Tensor B, int stride_d, int stride_h, int stride_w, int pad_t, int pad_h, int pad_w) -> Tensor");
    // set_tile_overrides takes NO tensor args, so a CUDA-specific impl would never
    // be selected (no tensor to anchor dispatch to the CUDA backend -> falls through
    // all backends -> RuntimeError). Register it as a catch-all (CompositeImplicitAutograd)
    // via m.def with a lambda so it runs regardless of dispatch key.
    m.def("set_tile_overrides(int s0, int s1, int s2) -> ()", [](int64_t s0, int64_t s1, int64_t s2) {
        g_tile_s0 = (int)s0; g_tile_s1 = (int)s1; g_tile_s2 = (int)s2;
    });
}
TORCH_LIBRARY_IMPL(fp16acc, CUDA, m) {
    m.impl("cutlass_conv3d", [](torch::Tensor A, torch::Tensor B,
                                int64_t sd, int64_t sh, int64_t sw) {
        return cutlass_conv3d_fp16acc(A, B, sd, sh, sw);
    });
    m.impl("cutlass_conv3d_padded", [](torch::Tensor A, torch::Tensor B,
                                       int64_t sd, int64_t sh, int64_t sw,
                                       int64_t pt, int64_t ph, int64_t pw) {
        return cutlass_conv3d_fp16acc_padded(A, B, sd, sh, sw, pt, ph, pw);
    });
}
// C++ Meta impl (torch.compile hot path uses this directly; Python register_fake
// route was 37ms slower per encode() due to Python callback overhead per conv).
TORCH_LIBRARY_IMPL(fp16acc, Meta, m) {
    m.impl("cutlass_conv3d", [](torch::Tensor A, torch::Tensor B,
                                int64_t sd, int64_t sh, int64_t sw) {
        int N = A.size(0), D = A.size(1), H = A.size(2), W = A.size(3);
        int K = B.size(0), T = B.size(1), R = B.size(2), S = B.size(3);
        int Z = (D - T) / (int)sd + 1, P = (H - R) / (int)sh + 1, Q = (W - S) / (int)sw + 1;
        return torch::empty({N, Z, P, Q, K}, A.options());
    });
    m.impl("cutlass_conv3d_padded", [](torch::Tensor A, torch::Tensor B,
                                       int64_t sd, int64_t sh, int64_t sw,
                                       int64_t pt, int64_t ph, int64_t pw) {
        int N = A.size(0), D = A.size(1), H = A.size(2), W = A.size(3);
        int K = B.size(0), T = B.size(1), R = B.size(2), S = B.size(3);
        int Z = (D + 2*(int)pt - T) / (int)sd + 1;
        int P = (H + 2*(int)ph - R) / (int)sh + 1;
        int Q = (W + 2*(int)pw - S) / (int)sw + 1;
        return torch::empty({N, Z, P, Q, K}, A.options());
    });
}

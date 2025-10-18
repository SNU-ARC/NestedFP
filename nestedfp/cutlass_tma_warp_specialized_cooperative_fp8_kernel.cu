#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
using TileSchedulerType    = cutlass::gemm::PersistentScheduler;

using namespace cute;

/* ==============================================================
 * 1.  Thread‑local CUDA‑graph‑safe context
 * ============================================================== */
template<int T1,int T2,int T3,int C1,int C2,int C3>
class TmaCoopFp8ScaleCtx_out {
 public:
  /* ---------- type aliases ---------------------------------- */
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          Shape<Int<T1>, Int<T2>, Int<T3>>,
          Shape<_1, _1, _1>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          /* α,β dtype */           float, float,
          /* C,D dtype/layout */    cutlass::half_t,  cutlass::layout::ColumnMajor, 8,
                                    cutlass::half_t,  cutlass::layout::ColumnMajor, 8,
          EpilogueScheduleType>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          /* A,B dtype/layout */    cutlass::float_e4m3_t, cutlass::layout::RowMajor,    16,
                                    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
          /* Accum */               float,
          Shape<Int<T1>, Int<T2>, Int<T3>>,
          Shape<Int<C1>, Int<C2>, Int<C3>>,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel   = cutlass::gemm::kernel::GemmUniversal<
                         Shape<int,int,int,int>,
                         CollectiveMainloop,
                         CollectiveEpilogue,
                         TileSchedulerType>;
  using DeviceKernel = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ElementCompute = typename DeviceKernel::EpilogueOutputOp::ElementCompute;
  using StrideA = typename DeviceKernel::GemmKernel::StrideA;
  using StrideB = typename DeviceKernel::GemmKernel::StrideB;
  using StrideC = typename DeviceKernel::GemmKernel::StrideC;
  using StrideD = typename DeviceKernel::GemmKernel::StrideD;

  /* ---------- (re)initialise --------------------------------*/
  void initialize(const at::Tensor& A,
                  const at::Tensor& B,
                  const at::Tensor& D) {
    query_shapes(A,B,D);
    make_workspace(A.device());
    build_args(A,B,D);
    initialised = true;
  }

  void maybe_reinit(const at::Tensor& A,
                    const at::Tensor& B,
                    const at::Tensor& D) {
    if (!initialised || !same_shape(A,B,D))
      initialize(A,B,D);
  }

  /* ---------- run -------------------------------------------*/
  void run(cudaStream_t stream,
           const at::Tensor& A,
           const at::Tensor& B,
           const at::Tensor& D) {
    build_args(A,B,D);   // only ptr refresh
    auto st = gemm.run(args, workspace.data_ptr<uint8_t>(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "CUTLASS cooperative FP8 GEMM failed");
  }

 private:
  /* ---------- helpers ---------------------------------------*/
  void query_shapes(const at::Tensor& A,
                    const at::Tensor& B,
                    const at::Tensor&) {
    M = static_cast<int>(A.size(0));
    N = static_cast<int>(B.size(0));
    K = static_cast<int>(A.size(1));
  }

  void make_workspace(const at::Device& dev) {
    size_t bytes = DeviceKernel::get_workspace_size(
                     typename DeviceKernel::Arguments{});
    workspace = at::empty({static_cast<long>(bytes)},
                          at::dtype(at::kByte).device(dev));
  }

  void build_args(const at::Tensor& A,
                  const at::Tensor& B,
                  const at::Tensor& D) {
    auto A_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(
                   A.contiguous().data_ptr());
    auto B_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(
                   B.contiguous().data_ptr());
    auto D_ptr = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

    int dev; cudaGetDevice(&dev);
    hw.device_id = dev;
    hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

    constexpr float kAlpha = 0.00390625f;   // 1/256
    constexpr float kBeta  = 0.f;

    /* --- build_args 수정 부분 ----------------------------------- */
    args = typename DeviceKernel::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, /*L*/1},

      /* A, B ----------------------------------------------------- */
      {
        A_ptr,
        /* was: make_cute_packed_stride(...) */
        cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1)),
        B_ptr,
        cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1))
      },

      /* epilogue ------------------------------------------------- */
      {
        {ElementCompute(kAlpha), ElementCompute(kBeta)},
        nullptr,
        cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1)),
        D_ptr,
        cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1))
      },

      hw
    };
  }

  bool same_shape(const at::Tensor& A,
                  const at::Tensor& B,
                  const at::Tensor& D) const {
    return A.size(0)==M && B.size(0)==N && A.size(1)==K &&
           D.size(0)==M && D.size(1)==N;
  }

  /* ---------- data ------------------------------------------*/
  bool                              initialised{false};
  int                               M{0}, N{0}, K{0};
  at::Tensor                        workspace;
  typename DeviceKernel::Arguments  args;
  DeviceKernel                      gemm;
  cutlass::KernelHardwareInfo       hw;
};

/* ==============================================================
 * 2.  PyBind‑visible *_out wrapper
 * ============================================================== */
template<int T1,int T2,int T3,int C1,int C2,int C3>
void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel(
        const at::Tensor& A,
        const at::Tensor& B,
              at::Tensor& D) {

  TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda() && D.device().is_cuda(),
              "All tensors must be on the same CUDA device");
#if defined(torch_kFloat8_e4m3fn)
  TORCH_CHECK(A.scalar_type()==torch::kFloat8_e4m3fn &&
              B.scalar_type()==torch::kFloat8_e4m3fn,
              "A,B must be torch.float8_e4m3fn");
#endif
  TORCH_CHECK(D.scalar_type()==torch::kF16,
              "D must be torch.float16");
  TORCH_CHECK(A.size(1)==B.size(1), "K mismatch (A.cols vs B.cols)");
  TORCH_CHECK(D.is_contiguous(),    "D must be contiguous");
  TORCH_CHECK(D.size(0)==A.size(0) && D.size(1)==B.size(0),
              "D shape must be (M,N)");

  static thread_local TmaCoopFp8ScaleCtx_out<
      T1,T2,T3,C1,C2,C3> ctx;

  ctx.maybe_reinit(A,B,D);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ctx.run(stream, A, B, D);
}

template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 16, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 16, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 16, 512, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 32, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 32, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 32, 512, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 64, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 64, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 64, 512, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 128, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 128, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 256, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<128, 256, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 16, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 16, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 32, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 32, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 64, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 64, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 128, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 128, 256, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_fp8_scale_kernel<256, 256, 128, 2, 1, 1>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);

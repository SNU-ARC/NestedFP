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

using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
using TileSchedulerType = cutlass::gemm::PersistentScheduler;

using namespace cute;

/* =========================================================
 * 1.  CUDA‑graph–safe context (out‑variant)
 *      ‑ caller supplies tensor D
 * ===================================================== */
template<int T1, int T2, int T3>
class TmaCtx_out {
 public:
  /* ---------- type aliases ----------------------------------- */
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          cute::Shape<cute::Int<T1>, cute::Int<T2>, cute::Int<T3>>,
          cute::Shape<cute::_1, cute::_1, cute::_1>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          float, float,
          cutlass::half_t, cutlass::layout::ColumnMajor, 8,
          cutlass::half_t, cutlass::layout::ColumnMajor, 8,
          EpilogueScheduleType>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          cutlass::half_t, cutlass::layout::RowMajor,    8,
          cutlass::half_t, cutlass::layout::ColumnMajor, 8,
          float,
          cute::Shape<cute::Int<T1>, cute::Int<T2>, cute::Int<T3>>,
          cute::Shape<cute::_1,  cute::_1,  cute::_1>,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel     = cutlass::gemm::kernel::GemmUniversal<
                           cute::Shape<int,int,int,int>,
                           CollectiveMainloop,
                           CollectiveEpilogue,
                           TileSchedulerType>;
  using DeviceKernel   = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ElementCompute = typename DeviceKernel::EpilogueOutputOp::ElementCompute;

  using StrideA = typename DeviceKernel::GemmKernel::StrideA;
  using StrideB = typename DeviceKernel::GemmKernel::StrideB;
  using StrideC = typename DeviceKernel::GemmKernel::StrideC;
  using StrideD = typename DeviceKernel::GemmKernel::StrideD;

  /* ---------- public API ------------------------------------- */
  void initialize(const at::Tensor& A,
                  const at::Tensor& B,
                  const at::Tensor& D) {
    queryShapes(A,B,D);
    createWorkspace(A.device());     // workspace only
    buildArguments(A,B,D);
    initialized = true;
  }

  /* 재초기화 ― shape(M,N,K) 변동 시 */
  void maybe_reinitialize(const at::Tensor& A,
                          const at::Tensor& B,
                          const at::Tensor& D) {
    if (!initialized || !sameShape(A,B,D)) {
      initialize(A,B,D);
    }
  }

  /* 단순 ptr 갱신 */
  void refresh_arguments(const at::Tensor& A,
                         const at::Tensor& B,
                         const at::Tensor& D) {
    buildArguments(A,B,D);
  }

  void run(cudaStream_t stream,
           const at::Tensor& A,
           const at::Tensor& B,
           const at::Tensor& D) {
    refresh_arguments(A,B,D);
    auto status = gemm_op.run(args,
                              workspace.data_ptr<uint8_t>(),
                              stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS GEMM failed");
  }

  bool sameShape(const at::Tensor& A,
                 const at::Tensor& B,
                 const at::Tensor& D) const {
    return (A.size(0)==M && B.size(0)==N &&
            A.size(1)==K &&        // K
            D.size(0)==M && D.size(1)==N);
  }

 private:
  /* ---------- helpers ---------------------------------------- */
  void queryShapes(const at::Tensor& A,
                   const at::Tensor& B,
                   const at::Tensor& /*D*/) {
    M = static_cast<int>(A.size(0));
    N = static_cast<int>(B.size(0));
    K = static_cast<int>(A.size(1));
  }

  void createWorkspace(const at::Device& dev) {
    size_t ws_bytes =
        DeviceKernel::get_workspace_size(
            typename DeviceKernel::Arguments{});
    workspace = at::empty({static_cast<long>(ws_bytes)},
                          at::dtype(at::kByte).device(dev));
  }

  void buildArguments(const at::Tensor& A,
                      const at::Tensor& B,
                      const at::Tensor& D) {
    const cutlass::half_t* A_ =
        reinterpret_cast<const cutlass::half_t*>(A.contiguous().data_ptr());
    const cutlass::half_t* B_ =
        reinterpret_cast<const cutlass::half_t*>(B.contiguous().data_ptr());
    cutlass::half_t* D_ =
        reinterpret_cast<cutlass::half_t*>(D.data_ptr());

    int dev_id; cudaGetDevice(&dev_id);
    hw_info.device_id = dev_id;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev_id);

    args = typename DeviceKernel::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, /*L=*/1},

      /* A, B (packed row/col) */
      {
        A_,
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1)),
        B_,
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1))
      },

      /* epilogue (C=null, α=1, β=0) */
      {
        {ElementCompute(1.f), ElementCompute(0.f)},
        nullptr,
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1)),
        D_,
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1))
      },

      hw_info
    };
  }

  /* ---------- data members ----------------------------------- */
  bool                         initialized{false};
  int                          M{0}, N{0}, K{0};
  at::Tensor                   workspace;
  typename DeviceKernel::Arguments args;
  DeviceKernel                 gemm_op;
  cutlass::KernelHardwareInfo  hw_info;
};

/* =========================================================
 * 2.  PyBind 노출 함수 ( *_out 스타일)
 *      ‑ returns void, writes into caller‑supplied D
 * ===================================================== */
template<int T1,int T2,int T3>
void cutlass_tma_warp_specialized_kernel(const at::Tensor& A,
                                             const at::Tensor& B,
                                            at::Tensor& D) {
  TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda() && D.device().is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(A.dtype() == torch::kF16 && B.dtype() == torch::kF16 && D.dtype() == torch::kF16,
              "All tensors must be torch.float16");
  TORCH_CHECK(A.size(1) == B.size(1), "K dimension mismatch (A.cols vs B.cols)");
  TORCH_CHECK(D.is_contiguous(),      "Output D must be contiguous");
  TORCH_CHECK(D.size(0) == A.size(0) && D.size(1) == B.size(0),
              "D must have shape (M, N)");

  /* thread‑local ctx: host‑thread당 하나 → graph‑safe */
  static thread_local TmaCtx_out<T1,T2,T3> ctx;

  ctx.maybe_reinitialize(A,B,D);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ctx.run(stream, A, B, D);
}

template void cutlass_tma_warp_specialized_kernel<64, 16, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 16, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 16, 256>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 32, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 32, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 32, 256>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 64, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 64, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 64, 256>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 128, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 128, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 128, 256>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 256, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<64, 256, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 16, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 16, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 16, 256>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 32, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 32, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 32, 256>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 64, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 64, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 64, 256>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 128, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 128, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 256, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<128, 256, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 16, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 16, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 32, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 32, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 64, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 64, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 128, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 128, 128>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_kernel<256, 256, 64>(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);

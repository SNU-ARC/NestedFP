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

//-----------------------------------------------------------------------
// 0.  Schedule / Types  (Cooperative, NO‑Padding)
//-----------------------------------------------------------------------
using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperativeCustom;
using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
using TileSchedulerType    = cutlass::gemm::PersistentScheduler;

using namespace cute;

//-----------------------------------------------------------------------
// 1.  CUDA‑graph–safe context  (out‑variant, NO padding)
//-----------------------------------------------------------------------

template<int T1,int T2,int T3,int C1,int C2,int C3>
class TmaCoopDualCtxNoPad {
 public:
  /* ---- type aliases ------------------------------------------------ */
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          cute::Shape<cute::Int<T1>, cute::Int<T2>, cute::Int<T3>>,
          cute::Shape<cute::_1,       cute::_1,       cute::_1>,
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
          cute::Shape<cute::Int<T1>, cute::Int<T2>,  cute::Int<T3>>,
          cute::Shape<cute::Int<C1>, cute::Int<C2>,  cute::Int<C3>>,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
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

  /* ---- public API -------------------------------------------------- */
  void maybe_reinitialize(const at::Tensor& A1,
                          const at::Tensor& B,
                          const at::Tensor& D) {
    if (!initialized_ || !sameShape(A1,B,D)) initialize(A1,B,D);
  }

  void run(cudaStream_t stream,
           const at::Tensor& A1,
           const at::Tensor& A2,
           const at::Tensor& B,
           const at::Tensor& D) {
    buildArguments(A1,A2,B,D);
    auto st = gemm_op_.run(args_, workspace_.data_ptr<uint8_t>(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS cooperative GEMM failed");
  }

 private:
  void initialize(const at::Tensor& A1,
                  const at::Tensor& B,
                  const at::Tensor& D) {
    queryShapes(A1,B,D);
    allocateWorkspace(A1.device());
    buildArguments(A1,A1,B,D);
    initialized_ = true;
  }

  void queryShapes(const at::Tensor& A1,
                   const at::Tensor& B,
                   const at::Tensor& /*D*/) {
    M_ = static_cast<int>(A1.size(0));
    N_ = static_cast<int>(B.size(0));
    K_ = static_cast<int>(A1.size(1));
  }

  bool sameShape(const at::Tensor& A1,
                 const at::Tensor& B,
                 const at::Tensor& D) const {
    return A1.size(0)==M_ && B.size(0)==N_ && A1.size(1)==K_ &&
           D.size(0)==M_ && D.size(1)==N_;
  }

  void allocateWorkspace(const at::Device& dev) {
    std::size_t ws = DeviceKernel::get_workspace_size(typename DeviceKernel::Arguments{});
    workspace_ = at::empty({static_cast<long>(ws)}, at::dtype(at::kByte).device(dev));
  }

  void buildArguments(const at::Tensor& A1,
                      const at::Tensor& A2,
                      const at::Tensor& B,
                      const at::Tensor& D) {
    const auto* A1_ = reinterpret_cast<const cutlass::float_e4m3_t*>(A1.contiguous().data_ptr());
    const auto* A2_ = reinterpret_cast<const cutlass::float_e4m3_t*>(A2.contiguous().data_ptr());
    const auto* B_  = reinterpret_cast<const cutlass::half_t*>(B.contiguous().data_ptr());
    auto*       D_  = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

    int dev; cudaGetDevice(&dev);
    hw_info_.device_id = dev;
    hw_info_.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

    args_ = typename DeviceKernel::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M_, N_, K_, 1},
      { A1_, A2_,
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_, K_, 1)),
        B_,  cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N_, K_, 1)) },
      { {ElementCompute(1.f), ElementCompute(0.f)},
        nullptr,
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_, N_, 1)),
        D_,  cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M_, N_, 1)) },
      hw_info_ };
  }

  bool                         initialized_ = false;
  int                          M_{0}, N_{0}, K_{0};
  at::Tensor                   workspace_;
  typename DeviceKernel::Arguments args_;
  DeviceKernel                 gemm_op_;
  cutlass::KernelHardwareInfo  hw_info_;
};

//-----------------------------------------------------------------------
// 2.  PyBind Entry  (No‑padding variant)
//-----------------------------------------------------------------------

template<int T1,int T2,int T3,int C1,int C2,int C3>
void cutlass_tma_warp_specialized_cooperative_custom_kernel(
        const at::Tensor& A1,
        const at::Tensor& A2,
        const at::Tensor& B,
        at::Tensor&       D) {
  TORCH_CHECK(A1.device().is_cuda() && A2.device().is_cuda() &&
              B.device().is_cuda()  && D.device().is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(A1.dtype()==torch::kFloat8_e4m3fn && A2.dtype()==torch::kFloat8_e4m3fn,
              "A1/A2 must be FP8 (e4m3)");
  TORCH_CHECK(B.dtype()==torch::kF16 && D.dtype()==torch::kF16,
              "B and D must be FP16");
  TORCH_CHECK(A1.sizes()==A2.sizes(), "A1/A2 shape mismatch");
  TORCH_CHECK(A1.size(1)==B.size(1),  "K dim mismatch");
  TORCH_CHECK(D.is_contiguous(),      "D must be contiguous");
  TORCH_CHECK(D.size(0)==A1.size(0) && D.size(1)==B.size(0),
              "D must have shape (M, N)");

  static thread_local TmaCoopDualCtxNoPad<T1,T2,T3,C1,C2,C3> ctx;
  ctx.maybe_reinitialize(A1,B,D);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ctx.run(stream,A1,A2,B,D);
}

template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 16, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 16, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 16, 256, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 32, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 32, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 32, 256, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 64, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 64, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 64, 256, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 128, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 128, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 256, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<128, 256, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 16, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 16, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 32, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 32, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 64, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 64, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 128, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 128, 128, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);
template void cutlass_tma_warp_specialized_cooperative_custom_kernel<256, 256, 64, 2, 1, 1>(const at::Tensor& A1, const at::Tensor& A2, const at::Tensor& B, at::Tensor& D);

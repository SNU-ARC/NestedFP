#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/util/host_tensor.h"

/*
using Type = cutlass::float_e4m3_t;
using ElementA = Type;
using ElementB = Type;
using ElementOutput = cutlass::half_t;
using ElementAuxOutput = ElementOutput;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;
*/

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    cutlass::half_t,
    cutlass::half_t,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,
    float,
    float
    >;

using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3, 16, 16, //kStages, kAlignmentA, kAlignmentB, 
    cutlass::arch::OpMultiplyAdd
  >;

bool e4m3(cutlass::float_e4m3_t* A, cutlass::float_e4m3_t* B, half* D, int M, int N, int K)
{
    typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
      cutlass::half_t(1.0),
      cutlass::half_t(0.0)
    };

    typename Gemm::EpilogueOutputOp::Params epilogue_params{
      activation_params,
      NULL,//scale_A.device_data(),
      NULL,//scale_B.device_data(),
      NULL,//scale_C.device_data(),
      NULL,//scale_D.device_data(),
      NULL,//scale_Aux.device_data(),
      NULL,//abs_max_Aux.device_data(),
      NULL,//abs_max_D.device_data()
    };

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord{M, N, K},//problem_size,
      // Batch count
      1,
      epilogue_params,
      A,//tensor_A.device_data(),
      B,//tensor_B.device_data(),
      NULL,//tensor_C.device_data(),
      D,//tensor_D.device_data(),
      NULL,//tensor_Aux.device_data(),
      NULL,//tensor_Vector.device_data(),
      // Batch strides
      M * K,//problem_size.m() * problem_size.k(),
      N * K,//problem_size.n() * problem_size.k(),
      M * N,//problem_size.m() * problem_size.n(),
      M * N,//problem_size.m() * problem_size.n(),
      M,//(int)problem_size.m(), // Vector
      // Leading dimension
      K,//problem_size.k(),//tensor_A.layout().stride(0),
      K,//problem_size.k(),//tensor_B.layout().stride(0),
      N,//problem_size.n(),//tensor_C.layout().stride(0),
      N,//problem_size.n(),//tensor_D.layout().stride(0),
      (int64_t)0
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    status = gemm_op();
  
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
      return false;
    }
    return true;
}


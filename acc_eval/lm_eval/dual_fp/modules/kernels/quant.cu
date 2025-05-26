#include <assert.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "e4m3.h"
#include "e5m2.h"
#include "e4m3_scale.h"

void gemm_e4m3(torch::Tensor X, torch::Tensor W, torch::Tensor Y) {
	cutlass::float_e4m3_t* X_ = reinterpret_cast<cutlass::float_e4m3_t*>(X.data_ptr());
	cutlass::float_e4m3_t* W_ = reinterpret_cast<cutlass::float_e4m3_t*>(W.data_ptr());
	half* Y_ = reinterpret_cast<half*>(Y.data_ptr());

	e4m3(X_, W_, Y_, X.size(0), W.size(1), X.size(1));
}

void gemm_e5m2(torch::Tensor X, torch::Tensor W, torch::Tensor Y) {
        cutlass::float_e5m2_t* X_ = reinterpret_cast<cutlass::float_e5m2_t*>(X.data_ptr());
        cutlass::float_e5m2_t* W_ = reinterpret_cast<cutlass::float_e5m2_t*>(W.data_ptr());
        half* Y_ = reinterpret_cast<half*>(Y.data_ptr());

        e5m2(X_, W_, Y_, X.size(0), W.size(1), X.size(1));
}

void gemm_e4m3_scale(torch::Tensor X, torch::Tensor W, torch::Tensor Y) {
        cutlass::float_e4m3_t* X_ = reinterpret_cast<cutlass::float_e4m3_t*>(X.data_ptr());
        cutlass::float_e4m3_t* W_ = reinterpret_cast<cutlass::float_e4m3_t*>(W.data_ptr());
        half* Y_ = reinterpret_cast<half*>(Y.data_ptr());

        e4m3_scale(X_, W_, Y_, X.size(0), W.size(1), X.size(1));
}

__global__ void E4M3_ROUND_TO_NEAREST(half* SRC, cutlass::float_e4m3_t* DEST, int S) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= S) return;

        DEST[tid] = cutlass::float_e4m3_t::from_half(SRC[tid]);
}

void quant_e4m3(torch::Tensor A, torch::Tensor A_) {
        int D1 = A.sizes()[0];
        int D2 = A.sizes()[1];

        half* SRC_A = reinterpret_cast<half*>(A.data_ptr());
        cutlass::float_e4m3_t* DEST_A = reinterpret_cast<cutlass::float_e4m3_t*>(A_.data_ptr());

        int BLK_SZ = 256;
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = BLK_SZ;
        gridDim.x = (D1 * D2 + BLK_SZ - 1) / BLK_SZ;
        E4M3_ROUND_TO_NEAREST<<<gridDim, blockDim>>>(SRC_A, DEST_A, D1 * D2);
}

__global__ void E5M2_ROUND_TO_NEAREST(half* SRC, cutlass::float_e5m2_t* DEST, int S) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= S) return;

        DEST[tid] = cutlass::float_e5m2_t::from_half(SRC[tid]);
}

void quant_e5m2(torch::Tensor A, torch::Tensor A_) {
        int D1 = A.sizes()[0];
        int D2 = A.sizes()[1];

        half* SRC_A = reinterpret_cast<half*>(A.data_ptr());
        cutlass::float_e5m2_t* DEST_A = reinterpret_cast<cutlass::float_e5m2_t*>(A_.data_ptr());

        int BLK_SZ = 256;
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = BLK_SZ;
        gridDim.x = (D1 * D2 + BLK_SZ - 1) / BLK_SZ;
        E5M2_ROUND_TO_NEAREST<<<gridDim, blockDim>>>(SRC_A, DEST_A, D1 * D2);
}

__global__ void FP8_ROUND_TOWARD_ZERO(half* SRC, cutlass::float_e4m3_t* DEST, int S) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= S) return;

	half h = SRC[tid];
	uint16_t a = reinterpret_cast<uint16_t const &>(h);
	uint8_t s1 = (a >> 15) & 0x1;
	uint8_t e1 = (a >> 10) & 0xf;
	uint8_t m1 = (a >> 7) & 0x7;
	uint8_t m2 = a & 0x7f;
	uint8_t b = (s1 << 7) | (e1 << 3) | (m1 << 0);

	assert(((a >> 14) & 0x1) == 0);

	if (e1 == 15 && m1 == 6) {
		DEST[tid] = cutlass::float_e4m3_t::bitcast(b);
		return;
	}

	if ((m2 > 64) || ((m2 == 64) && ((m1 & 0x1) == 1))) DEST[tid] = cutlass::float_e4m3_t::bitcast(b + 1);
	else DEST[tid] = cutlass::float_e4m3_t::bitcast(b);
}

void quant_e4m3_scale(torch::Tensor A, torch::Tensor A_) {
        int D1 = A.sizes()[0];
	int D2 = A.sizes()[1];

        half* SRC_A = reinterpret_cast<half*>(A.data_ptr());
        cutlass::float_e4m3_t* DEST_A = reinterpret_cast<cutlass::float_e4m3_t*>(A_.data_ptr());
        
        int BLK_SZ = 256;
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = BLK_SZ;
        gridDim.x = (D1 * D2 + BLK_SZ - 1) / BLK_SZ;
        FP8_ROUND_TOWARD_ZERO<<<gridDim, blockDim>>>(SRC_A, DEST_A, D1 * D2);
}

PYBIND11_MODULE(dual_fp_ext, m) {
    m.def("gemm_e4m3", &gemm_e4m3);
    m.def("gemm_e5m2", &gemm_e5m2);
    m.def("gemm_e4m3_scale", &gemm_e4m3_scale);
    m.def("quant_e4m3", &quant_e4m3);
    m.def("quant_e5m2", &quant_e5m2);
    m.def("quant_e4m3_scale", &quant_e4m3_scale);
}

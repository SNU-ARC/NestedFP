#include <torch/extension.h>
#include "cutlass/float8.h"
#include "cutlass/half.h"

__global__ void divide_fp16_kernel(cutlass::half_t* S, cutlass::float_e4m3_t* D1, cutlass::float_e4m3_t* D2, int N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= N) return;

        uint16_t a = S[tid].raw();
        uint8_t b = a & 0xff;
        D2[tid] = cutlass::float_e4m3_t::bitcast(b);

        uint8_t s = (a >> 15) & 0x1;
        uint8_t e = (a >> 10) & 0xf;
        uint8_t m1 = (a >> 7) & 0x7;
        uint8_t m2 = a & 0x7f;
        uint8_t c = (s << 7) | (e << 3) | (m1 << 0);

	assert(((a >> 14) & 0x1) == 0);

        if (e == 15 && m1 == 6) {
                D1[tid] = cutlass::float_e4m3_t::bitcast(c);
                return;
        }

        if ((m2 > 64) || ((m2 == 64) && ((m1 & 0x1) == 1))) D1[tid] = cutlass::float_e4m3_t::bitcast(c + 1);
        else D1[tid] = cutlass::float_e4m3_t::bitcast(c);
}

__global__ void merge_fp8_kernel(cutlass::float_e4m3_t* S1, cutlass::float_e4m3_t* S2, cutlass::half_t* D, int N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= N) return;

        uint8_t a = S1[tid].raw();
        uint8_t b = S2[tid].raw();
        uint8_t s = a & 0x80;
        uint8_t sub = (a & 0x1) ^ (b >> 7);
        uint8_t e_ = ((a - sub) >> 1) & 0x3f;
        
        D[tid] = cutlass::half_t::bitcast(((s | e_) << 8) | b);
}

__global__ void naive_divide_fp16_kernel(cutlass::half_t* S, cutlass::float_e4m3_t* D1, cutlass::float_e4m3_t* D2, int N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= N) return;

        uint16_t a = S[tid].raw();
        uint8_t b = a & 0xff;
        D2[tid] = cutlass::float_e4m3_t::bitcast(b);
        uint8_t c = (a >> 8) & 0xff;
        D1[tid] = cutlass::float_e4m3_t::bitcast(c);
}

__global__ void naive_merge_fp8_kernel(cutlass::float_e4m3_t* S1, cutlass::float_e4m3_t* S2, cutlass::half_t* D, int N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= N) return;

        uint8_t a = S1[tid].raw();
        uint8_t b = S2[tid].raw();
        D[tid] = cutlass::half_t::bitcast((a << 8) | b);
}

void divide_fp16(const at::Tensor& S, const at::Tensor& D1, const at::Tensor& D2) {
	cutlass::half_t* S_ = reinterpret_cast<cutlass::half_t*>(S.data_ptr());
	cutlass::float_e4m3_t* D1_ = reinterpret_cast<cutlass::float_e4m3_t*>(D1.data_ptr());
	cutlass::float_e4m3_t* D2_ = reinterpret_cast<cutlass::float_e4m3_t*>(D2.data_ptr());

	int N = S.numel();
	int BLK_SZ = 256;
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = BLK_SZ;
        gridDim.x = (N + BLK_SZ - 1) / BLK_SZ;

	divide_fp16_kernel<<<gridDim, blockDim>>>(S_, D1_, D2_, N);
}

void merge_fp8(const at::Tensor& S1, const at::Tensor& S2, const at::Tensor& D) {
	cutlass::float_e4m3_t* S1_ = reinterpret_cast<cutlass::float_e4m3_t*>(S1.data_ptr());
        cutlass::float_e4m3_t* S2_ = reinterpret_cast<cutlass::float_e4m3_t*>(S2.data_ptr());
	cutlass::half_t* D_ = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

	int N = D.numel();
        int BLK_SZ = 256;
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = BLK_SZ;
        gridDim.x = (N + BLK_SZ - 1) / BLK_SZ;

	merge_fp8_kernel<<<gridDim, blockDim>>>(S1_, S2_, D_, N);
}

void naive_divide_fp16(const at::Tensor& S, const at::Tensor& D1, const at::Tensor& D2) {
        cutlass::half_t* S_ = reinterpret_cast<cutlass::half_t*>(S.data_ptr());
        cutlass::float_e4m3_t* D1_ = reinterpret_cast<cutlass::float_e4m3_t*>(D1.data_ptr());
        cutlass::float_e4m3_t* D2_ = reinterpret_cast<cutlass::float_e4m3_t*>(D2.data_ptr());

        int N = S.numel();
        int BLK_SZ = 128;
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = BLK_SZ;
        gridDim.x = (N + BLK_SZ - 1) / BLK_SZ;

        naive_divide_fp16_kernel<<<gridDim, blockDim>>>(S_, D1_, D2_, N);
}

void naive_merge_fp8(const at::Tensor& S1, const at::Tensor& S2, const at::Tensor& D) {
        cutlass::float_e4m3_t* S1_ = reinterpret_cast<cutlass::float_e4m3_t*>(S1.data_ptr());
        cutlass::float_e4m3_t* S2_ = reinterpret_cast<cutlass::float_e4m3_t*>(S2.data_ptr());
        cutlass::half_t* D_ = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

        int N = D.numel();
        int BLK_SZ = 128;
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = BLK_SZ;
        gridDim.x = (N + BLK_SZ - 1) / BLK_SZ;

        naive_merge_fp8_kernel<<<gridDim, blockDim>>>(S1_, S2_, D_, N);
}

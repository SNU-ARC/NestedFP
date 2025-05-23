#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <cstdio>
#include <torch/extension.h>

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

bool cublas_kernel(__half* A, __half* B, __half* D, int M, int N, int K) {
	float alpha = 1;
	float beta = 0;

	cublasHandle_t handle;
	CUBLAS_CHECK(cublasCreate(&handle));
	
	CUBLAS_CHECK(
		cublasGemmEx(handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			N, M, K,
			&alpha,
			B, CUDA_R_16F, K,
			A, CUDA_R_16F, K,
			&beta,
			D, CUDA_R_16F, N,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
		)
	);
	
	CUBLAS_CHECK(cublasDestroy(handle));
	return true;
}

bool cublas_tn(const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& Y) {
	half* X_ = reinterpret_cast<half*>(X.data_ptr());
	half* W_ = reinterpret_cast<half*>(W.data_ptr());
	half* Y_ = reinterpret_cast<half*>(Y.data_ptr());

	return cublas_kernel(X_, W_, Y_, X.size(0), W.size(0), X.size(1));
}

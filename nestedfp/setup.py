import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include = os.path.join(this_dir, "../cutlass/include")
cutlass_util_include = os.path.join(this_dir, "../cutlass/tools/util/include")

setup(
    name='cutlass',
    ext_modules=[
        CUDAExtension(
            'cutlass', 
            [
                'cutlass.cpp',
                'cutlass_tma_warp_specialized_kernel.cu',
                'cutlass_tma_warp_specialized_cooperative_kernel.cu',
                'cutlass_tma_warp_specialized_cooperative_streamk_kernel.cu',
                'cutlass_tma_warp_specialized_custom_kernel.cu',
                'cutlass_tma_warp_specialized_cooperative_custom_kernel.cu',
                'cutlass_tma_warp_specialized_cooperative_streamk_custom_kernel.cu',
                'cutlass_tma_warp_specialized_fp8_kernel.cu',
                'cutlass_tma_warp_specialized_cooperative_fp8_kernel.cu',
                'weight_handle.cu',
            ],
            include_dirs=[cutlass_include, cutlass_util_include],
            extra_compile_args={
                'nvcc': [
                    '-forward-unknown-to-host-compiler',
                    '-DCUTLASS_VERSIONS_GENERATED',
                    '-O3',
                    '-DNDEBUG',
                    '--generate-code=arch=compute_90a,code=[sm_90a]',
                    '--generate-code=arch=compute_90a,code=[compute_90a]',
                    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
                    '--expt-relaxed-constexpr',
                    '-DCUTE_USE_PACKED_TUPLE=1',
                    '-DCUTLASS_TEST_LEVEL=0',
                    '-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1',
                    '-DCUTLASS_DEBUG_TRACE_LEVEL=0',
                    '-Xcompiler=-fno-strict-aliasing',
                    '-std=c++17',
                    '-lineinfo',
                    '-Xptxas',
                    '-v',
                    '-D__CUDA_ARCH_FEAT_SM90_ALL',
                    '-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'
                ],
            },
            libraries=['cuda']
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

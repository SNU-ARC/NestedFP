from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="dual_fp_ext",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "dual_fp_ext", ["quant.cu"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

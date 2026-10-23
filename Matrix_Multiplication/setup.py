from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matmul_ext",
    ext_modules=[
        CUDAExtension(
            name="matmul_ext",
            sources=["matmul_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

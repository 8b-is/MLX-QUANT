from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add_one_ext",
    ext_modules=[
        CUDAExtension(
            name="add_one_ext",
            sources=["add_one_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

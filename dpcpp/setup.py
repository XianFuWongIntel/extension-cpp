from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

os.environ["CXX"] = "dpcpp"
setup(
    name='lltm_dpcpp',
    ext_modules=[
        CppExtension('lltm_dpcpp', ['lltm_dpcpp.cpp', 'lltm_dpcpp_kernel.dp.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

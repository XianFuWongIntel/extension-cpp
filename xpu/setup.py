from setuptools import setup
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name='lltm_xpu',
    ext_modules=[
        DPCPPExtension('lltm_xpu', ['lltm_xpu.cpp', 'lltm_xpu_kernel.dp.cpp']),
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    })

import intel_extension_for_pytorch
from torch.xpu.cpp_extension import load

lltm_xpu = load(
    'lltm_xpu', ['lltm_xpu.cpp', 'lltm_xpu_kernel.dp.cpp'], verbose=True)
help(lltm_xpu)
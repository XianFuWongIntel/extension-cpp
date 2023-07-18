from torch.utils.cpp_extension import load
import os

os.environ["CXX"] = "dpcpp"
lltm_dpcpp = load(
    'lltm_dpcpp', ['lltm_dpcpp.cpp', 'lltm_dpcpp_kernel.dp.cpp'], verbose=True)
help(lltm_dpcpp)

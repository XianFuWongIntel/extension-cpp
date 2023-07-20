from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

import python.lltm_baseline
import cpp.lltm


def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i))


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_forward(variables, options):
    baseline_values = python.lltm_baseline.LLTMFunction.apply(*variables)
    cpp_values = cpp.lltm.LLTMFunction.apply(*variables)

    print('Forward: Baseline (Python) vs. C++ ... ', end='')
    check_equal(baseline_values, cpp_values, options.verbose)
    print('Ok')

    if options.cuda:
        cuda_values = cuda.lltm.LLTMFunction.apply(*variables)
        print('Forward: Baseline (Python) vs. CUDA ... ', end='')
        check_equal(baseline_values, cuda_values, options.verbose)
        print('Ok')

    if options.dpcpp:
        dpcpp_values = dpcpp.lltm.LLTMFunction.apply(*variables)
        print('Forward: Baseline (Python) vs. DPC++ ... ', end='')
        check_equal(baseline_values, dpcpp_values, options.verbose)
        print('Ok')

def check_backward(variables, options):
    baseline_values = python.lltm_baseline.LLTMFunction.apply(*variables)
    (baseline_values[0] + baseline_values[1]).sum().backward()
    grad_baseline = get_grads(variables)

    zero_grad(variables)

    cpp_values = cpp.lltm.LLTMFunction.apply(*variables)
    (cpp_values[0] + cpp_values[1]).sum().backward()
    grad_cpp = get_grads(variables)

    print('Backward: Baseline (Python) vs. C++ ... ', end='')
    check_equal(grad_baseline, grad_cpp, options.verbose)
    print('Ok')

    if options.cuda:
        zero_grad(variables)
        cuda_values = cuda.lltm.LLTMFunction.apply(*variables)
        (cuda_values[0] + cuda_values[1]).sum().backward()
        grad_cuda = get_grads(variables)

        print('Backward: Baseline (Python) vs. CUDA ... ', end='')
        check_equal(grad_baseline, grad_cuda, options.verbose)
        print('Ok')

    if options.dpcpp:
        zero_grad(variables)
        dpcpp_values = dpcpp.lltm.LLTMFunction.apply(*variables)
        (dpcpp_values[0] + dpcpp_values[1]).sum().backward()
        grad_dpcpp = get_grads(variables)

        print('Backward: Baseline (Python) vs. DPC++ ... ', end='')
        check_equal(grad_baseline, grad_dpcpp, options.verbose)
        print('Ok')

    if options.xpu:
        zero_grad(variables)
        xpu_values = xpu.lltm.LLTMFunction.apply(*variables)
        (xpu_values[0] + xpu_values[1]).sum().backward()
        grad_xpu = get_grads(variables)

        print('Backward: Baseline (Python) vs. XPU ... ', end='')
        check_equal(grad_baseline, grad_xpu, options.verbose)
        print('Ok')

parser = argparse.ArgumentParser()
parser.add_argument('direction', choices=['forward', 'backward'], nargs='+')
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-dpc', '--dpcpp', action='store_true')
parser.add_argument('-x', '--xpu', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')
options = parser.parse_args()

if options.cuda:
    import cuda.lltm
    device = torch.device("cuda")
elif options.xpu:
    import intel_extension_for_pytorch
    import xpu.lltm
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
    if options.dpcpp:
        import dpcpp.lltm

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}
X = torch.randn(options.batch_size,
                options.features,
                **kwargs)
h = torch.randn(options.batch_size, options.state_size, **kwargs)
C = torch.randn(options.batch_size, options.state_size, **kwargs)
W = torch.randn(3 * options.state_size, options.features + options.state_size, **kwargs)
b = torch.randn(1, 3 * options.state_size, **kwargs)

variables = [X, W, b, h, C]

if 'forward' in options.direction:
    check_forward(variables, options)

if 'backward' in options.direction:
    check_backward(variables, options)

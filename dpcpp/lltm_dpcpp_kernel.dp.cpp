#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>

#include <vector>

namespace {
template <typename scalar_t> __dpct_inline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + sycl::exp(-z));
}

template <typename scalar_t> __dpct_inline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t> __dpct_inline__ scalar_t d_tanh(scalar_t z) {
  const auto t = sycl::tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__dpct_inline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  /*
  DPCT1064:4: Migrated fmaxf call is used in a macro definition and is not valid
  for all macro uses. Adjust the code.
  */
  return sycl::fmax((float)(0.0), (float)z) +
         sycl::fmin((float)(0.0), (float)(alpha * (sycl::exp(z) - 1.0)));
}

template <typename scalar_t>
__dpct_inline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = sycl::exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

template <typename scalar_t>
void lltm_dpcpp_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3> gates,
    const torch::PackedTensorAccessor32<scalar_t,2> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2> new_h,
    torch::PackedTensorAccessor32<scalar_t,2> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2> candidate_cell,
    const sycl::nd_item<3> &item_ct1) {
  //batch index
  const int n = item_ct1.get_group(1);
  // column index
  const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  if (c < gates.size(2)){
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = elu(gates[n][2][c]);
    new_cell[n][c] =
        old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
    new_h[n][c] = sycl::tanh(new_cell[n][c]) * output_gate[n][c];
  }
}

template <typename scalar_t>
void lltm_dpcpp_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2> d_old_cell,
    torch::PackedTensorAccessor32<scalar_t,3> d_gates,
    const torch::PackedTensorAccessor32<scalar_t,2> grad_h,
    const torch::PackedTensorAccessor32<scalar_t,2> grad_cell,
    const torch::PackedTensorAccessor32<scalar_t,2> new_cell,
    const torch::PackedTensorAccessor32<scalar_t,2> input_gate,
    const torch::PackedTensorAccessor32<scalar_t,2> output_gate,
    const torch::PackedTensorAccessor32<scalar_t,2> candidate_cell,
    const torch::PackedTensorAccessor32<scalar_t,3> gate_weights,
    const sycl::nd_item<3> &item_ct1) {
  //batch index
  const int n = item_ct1.get_group(1);
  // column index
  const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
  if (c < d_gates.size(2)){
    /*
    DPCT1064:5: Migrated tanh call is used in a macro definition and is not
    valid for all macro uses. Adjust the code.
    */
    const auto d_output_gate =
        sycl::tanh((double)(new_cell[n][c])) * grad_h[n][c];
    const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
    const auto d_new_cell =
        d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


    d_old_cell[n][c] = d_new_cell;
    const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
    const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

    d_gates[n][0][c] =
        d_input_gate * d_sigmoid(gate_weights[n][0][c]);
    d_gates[n][1][c] =
        d_output_gate * d_sigmoid(gate_weights[n][1][c]);
    d_gates[n][2][c] =
        d_candidate_cell * d_elu(gate_weights[n][2][c]);
  }
}
} // namespace

std::vector<torch::Tensor> lltm_dpcpp_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto gates = gate_weights.reshape({batch_size, 3, state_size});
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const sycl::range<3> blocks(1, batch_size,
                              (state_size + threads - 1) / threads);

  /*
  DPCT1038:0: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_dpcpp", ([&] {
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto gates_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct0 =
          gates
              .packed_accessor32<scalar_t, 3>();
      auto
          old_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct1 =
              old_cell.packed_accessor32<scalar_t, 2>();
      auto new_h_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct2 =
          new_h
              .packed_accessor32<scalar_t, 2>();
      auto
          new_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct3 =
              new_cell.packed_accessor32<scalar_t, 2>();
      auto
          input_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct4 =
              input_gate.packed_accessor32<scalar_t, 2>();
      auto
          output_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct5 =
              output_gate.packed_accessor32<scalar_t, 2>();
      auto
          candidate_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct6 =
              candidate_cell.packed_accessor32<scalar_t, 2>();

      cgh.parallel_for(
          sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, threads),
                            sycl::range<3>(1, 1, threads)),
          [=](sycl::nd_item<3> item_ct1) {
            lltm_dpcpp_forward_kernel<scalar_t>(
                gates_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct0,
                old_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct1,
                new_h_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct2,
                new_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct3,
                input_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct4,
                output_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct5,
                candidate_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct6,
                item_ct1);
          });
    });
                             }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

std::vector<torch::Tensor> lltm_dpcpp_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gates,
    torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const sycl::range<3> blocks(1, batch_size,
                              (state_size + threads - 1) / threads);

  /*
  DPCT1038:2: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_dpcpp", ([&] {
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto
          d_old_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct0 =
              d_old_cell.packed_accessor32<scalar_t, 2>();
      auto d_gates_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct1 =
          d_gates
              .packed_accessor32<scalar_t, 3>();
      auto grad_h_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct2 =
          grad_h
              .packed_accessor32<scalar_t, 2>();
      auto
          grad_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct3 =
              grad_cell.packed_accessor32<scalar_t, 2>();
      auto
          new_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct4 =
              new_cell.packed_accessor32<scalar_t, 2>();
      auto
          input_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct5 =
              input_gate.packed_accessor32<scalar_t, 2>();
      auto
          output_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct6 =
              output_gate.packed_accessor32<scalar_t, 2>();
      auto
          candidate_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct7 =
              candidate_cell.packed_accessor32<scalar_t, 2>();
      auto gates_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct8 =
          gates
              .packed_accessor32<scalar_t, 3>();

      cgh.parallel_for(
          sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, threads),
                            sycl::range<3>(1, 1, threads)),
          [=](sycl::nd_item<3> item_ct1) {
            lltm_dpcpp_backward_kernel<scalar_t>(
                d_old_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct0,
                d_gates_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct1,
                grad_h_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct2,
                grad_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct3,
                new_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct4,
                input_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct5,
                output_gate_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct6,
                candidate_cell_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct7,
                gates_packed_accessor_scalar_t_torch_RestrictPtrTraits_size_t_ct8,
                item_ct1);
          });
    });
                             }));

  auto d_gate_weights = d_gates.flatten(1, 2);
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}

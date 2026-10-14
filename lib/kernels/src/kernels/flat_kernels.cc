#include "kernels/flat_kernels.h"
#include "kernels/flat_kernels_cpu.h"
#include "kernels/flat_kernels_gpu.h"

namespace FlexFlow {

void flat_forward_kernel(device_stream_t const &stream,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &output) {

  if (stream.is_gpu()) {
    DataType datatype =
        require_same(input.shape.data_type, output.shape.data_type);

    positive_int num_elements =
        require_same(get_num_elements(input.shape.dims),
                     get_num_elements(output.shape.dims));

    flat_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*input_ptr=*/input.ptr,
        /*output_ptr=*/output.ptr,
        /*num_elements=*/num_elements.size_t_from_positive_int(),
        /*element_size_in_bytes=*/
        size_of_datatype(datatype).size_t_from_positive_int());

  } else {
    ASSERT(stream.is_cpu());
    flat_cpu_forward_kernel(
        /*input=*/input,
        /*output=*/output);
  }
}

void flat_backward_kernel(device_stream_t const &stream,
                          GenericTensorAccessorR const &output_grad,
                          GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    positive_int num_elements =
        require_same(get_num_elements(output_grad.shape.dims),
                     get_num_elements(input_grad.shape.dims));

    flat_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*output_grad_ptr=*/output_grad.get_float_ptr(),
        /*input_grad_ptr=*/input_grad.get_float_ptr(),
        /*num_elements=*/num_elements.size_t_from_positive_int());
  } else {
    ASSERT(stream.is_cpu());
    flat_cpu_backward_kernel(
        /*output_grad=*/output_grad,
        /*input_grad=*/input_grad);
  }
}

} // namespace FlexFlow

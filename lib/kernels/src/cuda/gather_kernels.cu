/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "device.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"
#include "kernels/gather_kernels.h"

namespace FlexFlow {
namespace Kernels {
namespace Gather {

 GatherPerDeviceState init_kernel(PerDeviceFFHandle handle, int legion_dim, DataType index_data_type) {
    GatherPerDeviceState per_device_state = {handle, legion_dim, index_data_type};
    return GatherPerDeviceState;
 }

void gather_forward(float const *input_ptr,
                    float const *index_ptr,
                    float *output_ptr,
                    size_t output_size,
                    size_t stride,
                    ff_dim_t dim) {
  CUDA_KERNEL_LOOP(o, output_size) {
    size_t outer_index = o / (stride * dim.value());
    size_t left_over = o % stride;
    size_t input_idx = outer_index * (stride * dim.value()) +
                       index_ptr[o] * stride + left_over;
    output_ptr[o] = input_ptr[input_idx];
  }
}

void gather_backward(float const *output_grad_ptr,
                     float const *index_ptr,
                     float *input_grad_ptr,
                     size_t output_size,
                     size_t stride,
                     ff_dim_t dim) {
  CUDA_KERNEL_LOOP(o, output_size) {
    size_t outer_index = o / (stride * dim.value());
    size_t left_over = o % stride;
    size_t input_idx = outer_index * (stride * dim.value()) +
                       index_ptr[o] * stride + left_over;
    input_grad_ptr[input_idx] = output_grad_ptr[o];
  }
}

void forward_kernel(cudaStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output,
                    size_t stride,
                    ff_dim_t dim) {
  gather_forward(input.get_float_ptr(),
                 index.get_float_ptr(),
                 output.get_float_ptr(),
                 output.shape.get_volume(),
                 stride,
                 dim);
}

void backward_kernel(cudaStream_t stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad,
                     size_t stride,
                     ff_dim_t dim) {
  gather_backward(output_grad.get_float_ptr(),
                  index.get_float_ptr(),
                  input_grad.get_float_ptr(),
                  output_grad.shape.get_volume(),
                  stride,
                  dim);
}

} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

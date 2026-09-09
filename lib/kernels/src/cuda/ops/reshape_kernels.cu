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

#include "internal/device.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/reshape_kernels_gpu.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

void reshape_gpu_forward_kernel(cudaStream_t stream,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  require_same(input.shape.data_type, output.shape.data_type);
  require_same(get_num_elements(input.shape.dims),
               get_num_elements(output.shape.dims));

  size_t num_bytes =
      get_size_in_bytes(output.shape).unwrap_num_bytes().unwrap_nonnegative();

  checkCUDA(cudaMemcpyAsync(
      output.ptr, input.ptr, num_bytes, cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
__global__ void reshape_accumulate_kernel(T *input_grad,
                                          T const *output_grad,
                                          size_t num_elements) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    input_grad[i] += output_grad[i];
  }
}

template <DataType DT>
struct ReshapeGPUBackwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) const {
    int num_elements =
        get_num_elements(input_grad.shape.dims).int_from_positive_int();

    reshape_accumulate_kernel<real_type_t<DT>>
        <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
            input_grad.get<DT>(), output_grad.get<DT>(), num_elements);
  }
};

void reshape_gpu_backward_kernel(cudaStream_t stream,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad) {
  DataType data_type =
      require_same(output_grad.shape.data_type, input_grad.shape.data_type);
  require_same(get_num_elements(output_grad.shape.dims),
               get_num_elements(input_grad.shape.dims));

  DataTypeDispatch1<ReshapeGPUBackwardKernel>{}(
      data_type, stream, output_grad, input_grad);
}

} // namespace FlexFlow

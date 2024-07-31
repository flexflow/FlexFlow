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
#include "kernels/partition_kernels.h"

namespace FlexFlow {
namespace Kernels {
namespace Repartition {

template <DataType T>
struct ForwardKernel {
  void operator()(cudaStream_t stream,
                  RepartitionPerDeviceState const &m,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    checkCUDA(cudaMemcpyAsync(output.get<T>(),
                              input.get<T>(),
                              input.shape.num_elements() * size_of_datatype(T),
                              cudaMemcpyDeviceToDevice,
                              stream));
  }
};

template <DataType T>
struct BackwardKernel {
  void operator()(cudaStream_t stream,
                  RepartitionPerDeviceState const &m,
                  GenericTensorAccessorW const &input_grad,
                  GenericTensorAccessorR const &output_grad) {
    add_kernel<real_type<T>><<<GET_BLOCKS(input_grad.shape.num_elements()),
                               CUDA_NUM_THREADS,
                               0,
                               stream>>>(input_grad.get<T>(),
                                         output_grad.get<T>(),
                                         input_grad.shape.num_elements());
  }
};

RepartitionPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                      DataType data_type) {
  RepartitionPerDeviceState per_device_state = {handle, data_type};
  return per_device_state;
}

void forward_kernel(cudaStream_t stream,
                    RepartitionPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(m.data_type, stream, m, input, output);
}

void backward_kernel(cudaStream_t stream,
                     RepartitionPerDeviceState const &m,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &output_grad) {
  DataTypeDispatch1<BackwardKernel>{}(
      m.data_type, stream, m, input_grad, output_grad);
}

} // namespace Repartition
} // namespace Kernels
} // namespace FlexFlow

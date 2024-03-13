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


#include "kernels/datatype_dispatch.h"
#include "kernels/reshape_kernels.h"

namespace FlexFlow {

ReshapePerDeviceState init_kernel(DataType data_type) {
  return ReshapePerDeviceState{data_type};
}

namespace Kernels {
namespace Reshape {

template <DataType T>
struct ForwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    checkCUDA(cudaMemcpyAsync(output.get<T>(),
                              input.get<T>(),
                              input.shape.num_elements() * sizeof(T),
                              cudaMemcpyDeviceToDevice,
                              stream));
  }
}

template <DataType T>
struct BackwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorW const &input,
                  GenericTensorAccessorR const &output) {
    float alpha = 1.0f;
    apply_add_with_scale<T><<<GET_BLOCKS(input.shape.num_elements()),
                              CUDA_NUM_THREADS,
                              0,
                              stream>>>(
        input.get<T>(), output.get<T>(), input.shape.num_elements(), (T)alpha);
  }
}

void forward_kernel(cudaStream_t stream,
                    ReshapePerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(m.data_type, stream, input, output);
}

void backward_kernel(cudaStream_t stream,
                     ReshapePerDeviceState const &m,
                     GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output) {
  DataTypeDispatch1<BackwardKernel>{}(m.data_type, stream, input, output);
}

} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

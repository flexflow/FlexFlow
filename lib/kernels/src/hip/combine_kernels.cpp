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

#include "kernels/combine_kernels.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

CombinePerDeviceState::CombinePerDeviceState(FFHandler handler)
    : PerDeviceOpState(handler) {}

namespace Kernels {
namespace Combine {

template <DataType DT>
struct ForwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    checkCUDA(hipMemcpyAsync(output.get<DT>(),
                             input.get<DT>(),
                             input.shape.get_volume() * size_of(DT),
                             hipMemcpyDeviceToDevice,
                             stream));
  }
};

template <DataType DT>
struct BackwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) {
    size_t num_elements = output_grad.shape.get_volume();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<T>),
                       GET_BLOCKS(num_elements),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       input_grad.get<DT>(),
                       output_grad.get<DT>(),
                       num_elements);
  }
};

void forward_kernel(ffStream_t stream,
                    CombinePerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(m->data_type, stream, input, output);
}

void backward_kernel(ffStream_t stream,
                     CombinePerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad) {
  DataTypeDispatch1<BackwardKernel>{}(
      m->data_type, stream, output_grad, input_grad);
}

} // namespace Combine
} // namespace Kernels
} // namespace FlexFlow

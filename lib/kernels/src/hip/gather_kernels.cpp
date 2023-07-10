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

#include "kernels/gather_kernels.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

GatherPerDeviceState::GatherPerDeviceState(FFHandler handler)
    : PerDeviceOpState(handler){};

namespace Kernels {
namespace Gather {

template <DataType IndexTxype>
struct ForwardKernel {
  void operator()(hipStream_t stream,
                  GatherPerDeviceState const *m,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorR const &index,
                  GenericTensorAccessorW const &output,
                  size_t stride,
                  size_t input_size,
                  size_t output_size) {
    handle_unimplemented_hip_kernel(OP_GATHER);
  }
};

void forward_kernel(hipStream_t stream,
                    GatherPerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output,
                    size_t stride,
                    size_t input_size,
                    size_t output_size) {
  DataTypeDispatch1<ForwardKernel>{}(m->index_data_type,
                                     stream,
                                     m,
                                     input,
                                     index,
                                     output,
                                     stride,
                                     input_size,
                                     output_size);
}

template <DataType IndexType>
struct BackwardKernel {
  void operator()(hipStream_t stream,
                  GatherPerDeviceState const *m,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorR const &index,
                  GenericTensorAccessorW const &input_grad,
                  size_t stride,
                  size_t input_size,
                  size_t output_size) {
    handle_unimplemented_hip_kernel(OP_GATHER);
  }
};

void backward_kernel(hipStream_t stream,
                     GatherPerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad,
                     size_t stride,
                     size_t input_size,
                     size_t output_size) {
  DataTypeDispatch1<BackwardKernel>{}(m->index_data_type,
                                      stream,
                                      m,
                                      output_grad,
                                      index,
                                      input_grad,
                                      stride,
                                      input_size,
                                      output_size);
}

} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

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
#include "kernels/cast_kernels.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {
namespace Kernels {
namespace Cast {

template <typename IDT, typename ODT>
__global__ void cast_forward(IDT const *input, ODT *output, size_t volume) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT)input[i];
  }
}

template <typename IDT, typename ODT>
__global__ void
    cast_backward(IDT const *input, ODT *output, size_t volume, ODT beta) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT)input[i] + beta * output[i];
  }
}

template <DataType IDT, DataType ODT>
struct ForwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = input.shape.get_volume();
    cast_forward<<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
        input.get<IDT>(), output.get<ODT>(), volume);
  }
};

template <DataType IDT, DataType ODT>
struct BackwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = input.shape.get_volume();
    cast_backward<<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
        input.get<IDT>(), output.get<ODT>(), volume, cast_to<ODT>(1.0f));
  }
};

void forward_kernel(PerDeviceFFHandle handle,
                    ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    DataType input_type,
                    DataType output_type) {
  DataTypeDispatch2<ForwardKernel>{}(
      input_type, output_type, stream, input, output);
}

void backward_kernel(PerDeviceFFHandle handle,
                     ffStream_t stream,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &output,
                     DataType input_type,
                     DataType output_type) {
  DataTypeDispatch2<BackwardKernel>{}(
      input_type, output_type, stream, input, output);
}

} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow

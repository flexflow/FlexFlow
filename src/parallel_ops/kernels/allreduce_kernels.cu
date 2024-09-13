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

#include "flexflow/parallel_ops/kernels/allreduce_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

AllReduceMeta::AllReduceMeta(FFHandler handle, AllReduce const *reduct)
    : OpMeta(handle) {}

namespace Kernels {
namespace AllReduce {

void forward_kernel_wrapper(AllReduceMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
#ifdef FF_USE_NCCL
  // ncclDataType_t nccl_data_type = ff_to_nccl_datatype(input.data_type);
  checkNCCL(ncclAllReduce(input.ptr,
                          output.ptr,
                          input.domain.get_volume(),
                          ncclFloat,
                          ncclSum,
                          m->handle.ncclComm,
                          stream));
#else
  assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
}

void backward_kernel_wrapper(AllReduceMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input_grad.data_type == output_grad.data_type);
  assert(input_grad.domain == output_grad.domain);
#ifdef FF_USE_NCCL
  // ncclDataType_t nccl_data_type = ff_to_nccl_datatype(input.data_type);
  // std::cout <<"input volume: " << input.domain.get_volume() << "\n";
  // print_tensor<float>((float*)input.ptr, 32, "input ptr");
  checkNCCL(ncclAllReduce(output_grad.ptr,
                          input_grad.ptr,
                          output_grad.domain.get_volume(),
                          ncclFloat,
                          ncclSum,
                          m->handle.ncclComm,
                          stream));
#else
  assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
}

} // namespace AllReduce
} // namespace Kernels
} // namespace FlexFlow

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

#include "flexflow/parallel_ops/kernels/parallel_identity_kernels.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

ParallelIdentityMeta::ParallelIdentityMeta(FFHandler handle,
                                           ParallelIdentity const *reduct)
    : OpMeta(handle, reduct) {}

namespace Kernels {
namespace ParallelIdentity {

void forward_kernel_wrapper(ParallelIdentityMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
  size_t data_size = data_type_size(input.data_type);
  // copy input to output
  checkCUDA(hipMemcpyAsync(output.ptr,
                           input.ptr,
                           input.domain.get_volume() * data_size,
                           hipMemcpyDeviceToDevice,
                           stream));
}

void backward_kernel_wrapper(ParallelIdentityMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  assert(false && "To be implemented");
}

void inference_kernel_wrapper(ParallelIdentityMeta const *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
  size_t hidden_dim_size = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  size_t num_elements = bc->num_active_tokens();
  size_t data_size = data_type_size(input.data_type);
  checkCUDA(hipMemcpyAsync(output.ptr,
                           input.ptr,
                           hidden_dim_size * num_elements * data_size,
                           hipMemcpyDeviceToDevice,
                           stream));
}

void peft_bwd_kernel_wrapper(ParallelIdentityMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input_grad.data_type == output_grad.data_type);
  assert(input_grad.domain == output_grad.domain);
  size_t hidden_dim_size =
      input_grad.domain.hi()[0] - input_grad.domain.lo()[0] + 1;
  size_t num_elements = bc->num_active_tokens() * hidden_dim_size;
#ifdef FF_USE_NCCL
  ncclDataType_t nccl_data_type = ff_to_nccl_datatype(input_grad.data_type);
  checkNCCL(ncclAllReduce(output_grad.ptr,
                          input_grad.ptr,
                          num_elements,
                          nccl_data_type,
                          ncclSum,
                          m->handle.ncclComm,
                          stream));
#else
  assert(false && "Must enable FF_USE_NCCL to use ParallelIdentity operators");
#endif
}

} // namespace ParallelIdentity
} // namespace Kernels
} // namespace FlexFlow

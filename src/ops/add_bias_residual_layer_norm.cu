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

#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/add_bias_residual_layer_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

AddBiasResidualLayerNormMeta::AddBiasResidualLayerNormMeta(
    FFHandler handle,
    AddBiasResidualLayerNorm const *ln,
    MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handle) {
  elementwise_affine = ln->elementwise_affine;
  use_bias = ln->use_bias;
  effective_batch_size = ln->effective_batch_size;
  effective_num_elements = ln->effective_num_elements;
  profiling = ln->profiling;
  eps = ln->eps;
  DataType data_type = ln->data_type;
  size_t totalSize = effective_batch_size * data_type_size(data_type) * 6;
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  mean_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  rstd_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  ds_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  db_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  scale_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  bias_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
}

AddBiasResidualLayerNormMeta::~AddBiasResidualLayerNormMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

/*static*/
template <typename T>
void AddBiasResidualLayerNorm::inference_kernel(
    AddBiasResidualLayerNormMeta const *m,
    T const *in_ptr,
    T *out_ptr,
    T const *gamma_ptr,
    T const *beta_ptr,
    cudaStream_t stream) {}

/*static*/
void AddBiasResidualLayerNorm::inference_kernel_wrapper(
    AddBiasResidualLayerNormMeta const *m,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW &output,
    GenericTensorAccessorR const &residual,
    GenericTensorAccessorR const &attn_bias,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    AddBiasResidualLayerNorm::inference_kernel<float>(
        m,
        input.get_float_ptr(),
        output.get_float_ptr(),
        gamma.get_float_ptr(),
        m->use_bias ? beta.get_float_ptr() : nullptr,
        stream);
  } else if (m->input_type[0] == DT_HALF) {
    AddBiasResidualLayerNorm::inference_kernel<half>(
        m,
        input.get_half_ptr(),
        output.get_half_ptr(),
        gamma.get_half_ptr(),
        m->use_bias ? beta.get_half_ptr() : nullptr,
        stream);
  } else {
    assert(false && "unsupport datatype in layernorm");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[AddBiasResidualLayerNorm] forward time (CF) = %.2fms\n", elapsed);
    // print_tensor<T>(in_ptr, 32, "[AddBiasResidualLayerNorm:forward:input]");
    // print_tensor<T>(out_ptr, 32,
    // "[AddBiasResidualLayerNorm:forward:output]");
  }
}

}; // namespace FlexFlow

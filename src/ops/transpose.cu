/* Copyright 2020 Facebook
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

#include "flexflow/ops/transpose.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;

void Transpose::init_meta(TransposeMeta *m, Domain const &in_domain, Domain const &out_domain) const
{
  for (int i = 0; i < out_domain.get_dim(); i++) {
    assert(out_domain.hi()[i] == in_domain.hi()[this->perm[i]]);
    assert(out_domain.lo()[i] == in_domain.lo()[this->perm[i]]);
  }
  m->num_dim = out_domain.get_dim();
  for (int i = 0; i < m->num_dim; i++)
    m->perm[i] = this->perm[i];
}

OpMeta* Transpose::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Transpose* transpose = (const Transpose*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());

  TransposeMeta* m = new TransposeMeta(handle);
  transpose->init_meta(m, in_domain, out_domain);
  m->profiling = transpose->profiling;
  return m;
}

struct TransposeStrides
{
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM], perm[MAX_TENSOR_DIM];
};

__global__
void transpose_simple_kernel(coord_t volume,
                             const float* in_ptr,
                             float* out_ptr,
                             const TransposeStrides info,
                             const float beta)
{
  CUDA_KERNEL_LOOP(o_idx, volume)
  {
    coord_t i_idx = 0;
    coord_t t = o_idx;
    for (int i = info.num_dim-1; i >= 0; i--) {
      coord_t ratio = t / info.out_strides[i];
      t -= ratio * info.out_strides[i];
      i_idx += ratio * info.in_strides[info.perm[i]];
    }
    out_ptr[o_idx] += out_ptr[o_idx] * beta + in_ptr[i_idx];
  }
}

/*static*/
void Transpose::forward_kernel(const TransposeMeta* m,
                               const float* input_ptr,
                               float* output_ptr,
                               Domain in_domain,
                               Domain out_domain,
                              cudaStream_t stream)
{
  TransposeStrides info;
  info.num_dim = out_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (in_domain.hi()[i] - in_domain.lo()[i] + 1);
    int out_dim_size = (out_domain.hi()[i] - out_domain.lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[i] = m->perm[i];
  }
  transpose_simple_kernel<<<GET_BLOCKS(out_domain.get_volume()), CUDA_NUM_THREADS, 0, stream>>>(
      out_domain.get_volume(), input_ptr, output_ptr, info, 0.0f/*beta*/);
}

__host__
void Transpose::forward_task(const Task* task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Transpose* transpose = (const Transpose*) task->args;
  const TransposeMeta* m = *((TransposeMeta**) task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_domain.get_dim(); i++) {
    assert(out_domain.hi()[i] == in_domain.hi()[m->perm[i]]);
    assert(out_domain.lo()[i] == in_domain.lo()[m->perm[i]]);
  }
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(m, in_ptr, out_ptr, in_domain, out_domain, stream);
}

/*static*/
void Transpose::backward_kernel(const TransposeMeta* m,
                                float* input_grad_ptr,
                                const float* output_grad_ptr,
                                Domain in_grad_domain,
                                Domain out_grad_domain,
                                cudaStream_t stream)
{
  TransposeStrides info;
  info.num_dim = in_grad_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (out_grad_domain.hi()[i] - out_grad_domain.lo()[i] + 1);
    int out_dim_size = (in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[m->perm[i]] = i;
  }
  transpose_simple_kernel<<<GET_BLOCKS(in_grad_domain.get_volume()), CUDA_NUM_THREADS, 0, stream>>>(
      in_grad_domain.get_volume(), output_grad_ptr, input_grad_ptr, info, 1.0f/*beta*/);
}

__host__
void Transpose::backward_task(const Task* task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Transpose* transpose = (const Transpose*) task->args;
  const TransposeMeta* m = *((TransposeMeta**) task->local_args);
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_grad_domain.get_dim(); i++) {
    assert(out_grad_domain.hi()[i] == in_grad_domain.hi()[m->perm[i]]);
    assert(out_grad_domain.lo()[i] == in_grad_domain.lo()[m->perm[i]]);
  }
  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(m, in_grad_ptr, out_grad_ptr, in_grad_domain, out_grad_domain, stream);
}

bool Transpose::measure_operator_cost(Simulator* sim,
                                      const ParallelConfig& pc,
                                      CostMetrics& cost_metrics) const
{
  TensorBase sub_input, sub_output;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  TransposeMeta *m = sim->transpose_meta;
  this->init_meta(m, sub_input.get_domain(), sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, sub_input.get_domain(), sub_output.get_domain(), stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert (input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, input_grad_ptr, output_grad_ptr, sub_input.get_domain(), sub_output.get_domain(), stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Transpose] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Transpose] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow
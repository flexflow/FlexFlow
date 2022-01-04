/* Copyright 2020 Stanford
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

#include "flexflow/ops/embedding.h"
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

__host__
OpMeta* Embedding::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  const Embedding* embed = (Embedding*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  EmbeddingMeta* m = new EmbeddingMeta(handle);
  m->input_data_type = embed->inputs[0]->data_type;
  m->profiling = embed->profiling;
  m->aggr = embed->aggr;
  return m;
}

template<typename TI>
__global__
void embed_forward_no_aggr(
    const TI* input,
    float* output,
    const float* embed,
    int out_dim,
    int batch_size)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    TI wordIdx = input[idx];
    output[i] = embed[wordIdx * out_dim + off];
  }
}


template<typename TI>
__global__
void embed_forward_with_aggr(
    const TI* input,
    float* output,
    const float* embed,
    int out_dim,
    int in_dim,
    int batch_size,
    AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      TI wordIdx = input[idx * in_dim + j];
      output[i] += embed[wordIdx * out_dim + off];
      if (aggr == AGGR_MODE_SUM) {
      } else {
        assert(aggr == AGGR_MODE_AVG);
        output[i] /= in_dim;
      }
    }
  }
}

template<typename TI>
__global__
void embed_backward_no_aggr(
    const TI* input,
    const float* output,
    float* embed,
    int out_dim,
    int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    int idx = i / out_dim;
    int off = i % out_dim;
    TI wordIdx = input[idx];
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
  }
}

template<typename TI>
__global__
void embed_backward_with_aggr(
    const TI* input,
    const float* output,
    float* embed,
    int out_dim,
    int in_dim,
    int batch_size,
    AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    int idx = i / out_dim;
    int off = i % out_dim;
    float gradient;
    if (aggr == AGGR_MODE_SUM) {
       gradient = output[i];
    } else {
      assert(aggr == AGGR_MODE_AVG);
      gradient = output[i] / in_dim;
    }
    for (int j = 0; j < in_dim; j++) {
      TI wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}

template<typename TI>
void Embedding::forward_kernel(const TI* input_ptr,
                               float *output_ptr,
                               float const *weight_ptr,
                               int in_dim,
                               int out_dim,
                               int batch_size,
                               AggrMode aggr,
                               int outputSize,
                               cudaStream_t stream)
{
  if (aggr == AGGR_MODE_NONE) {
    embed_forward_no_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_ptr,out_dim, batch_size);
  } else {
    embed_forward_with_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_ptr, out_dim, in_dim, batch_size, aggr);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): kernel
*/

void Embedding::forward_task(const Task*task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime)
{
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  if (m->input_data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

template<typename TI>
void Embedding::forward_task_with_type(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain kernel_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  if (m->aggr == AGGR_MODE_NONE) {
    assert(kernel_domain.get_dim() == 2);
    assert(input_domain.get_dim() + 1 == output_domain.get_dim());
    for (size_t i = 0; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_domain.hi()[i+1]);
      assert(input_domain.lo()[i] == output_domain.lo()[i+1]);
    }
    assert(kernel_domain.hi()[0] - kernel_domain.lo()[0]
        == output_domain.hi()[0] - output_domain.lo()[0]);
  } else {
    assert(kernel_domain.get_dim() == 2);
    assert(input_domain.get_dim() + 1 == output_domain.get_dim());
    for (size_t i = 1; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_domain.hi()[i]);
      assert(input_domain.lo()[i] == output_domain.lo()[i]);
    }
    assert(kernel_domain.hi()[0] - kernel_domain.lo()[0]
        == output_domain.hi()[0] - output_domain.lo()[0]);
  }
  const TI* input_ptr = helperGetTensorPointerRO<TI>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  const float* kernel_ptr = helperGetTensorPointerRO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int in_dim, out_dim, effective_batch_size;
  if (m->aggr == AGGR_MODE_NONE) {
    in_dim = 1;
    out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
    effective_batch_size = output_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  } else {
    in_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
    out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
    effective_batch_size = output_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  }
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel<TI>(input_ptr, output_ptr, kernel_ptr,
      in_dim, out_dim, effective_batch_size,
      m->aggr, output_domain.get_volume(), stream);

  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    //print_tensor<TI>(input_ptr, input_domain.get_volume(), "[Embedding:forward:input]");
    //print_tensor<float>(kernel_ptr, kernel_domain.get_volume(), "[Embedding:forward:weight]");
    //print_tensor<float>(output_ptr, output_domain.get_volume(), "[Embedding:forward:output]");
  }
}

template<typename TI>
void Embedding::backward_kernel(const TI *input_ptr,
                                float const *output_ptr,
                                float *weight_grad_ptr,
                                int in_dim,
                                int out_dim,
                                int batch_size,
                                AggrMode aggr,
                                int outputSize,
                                cudaStream_t stream)
{
  if (aggr == AGGR_MODE_NONE) {
    embed_backward_no_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_grad_ptr, out_dim, batch_size);
  } else {
    embed_backward_with_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_grad_ptr, out_dim, in_dim, batch_size, aggr);
  }
}

__host__
void Embedding::backward_task(const Task*task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime)
{
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  if (m->input_data_type == DT_INT32) {
    backward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    backward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

template<typename TI>
void Embedding::backward_task_with_type(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain kernel_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  if (m->aggr == AGGR_MODE_NONE) {
    assert(kernel_grad_domain.get_dim() == 2);
    assert(input_domain.get_dim() + 1 == output_grad_domain.get_dim());
    for (size_t i = 0; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_grad_domain.hi()[i+1]);
      assert(input_domain.lo()[i] == output_grad_domain.lo()[i+1]);
    }
    assert(kernel_grad_domain.hi()[0] - kernel_grad_domain.lo()[0]
        == output_grad_domain.hi()[0] - output_grad_domain.lo()[0]);
  } else {
    assert(kernel_grad_domain.get_dim() == 2);
    assert(input_domain.get_dim() + 1 == output_grad_domain.get_dim());
    for (size_t i = 1; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_grad_domain.hi()[i]);
      assert(input_domain.lo()[i] == output_grad_domain.lo()[i]);
    }
    assert(kernel_grad_domain.hi()[0] - kernel_grad_domain.lo()[0]
        == output_grad_domain.hi()[0] - output_grad_domain.lo()[0]);
  }
  const TI* input_ptr = helperGetTensorPointerRO<TI>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* output_grad_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float* kernel_grad_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int in_dim, out_dim, effective_batch_size;
  if (m->aggr == AGGR_MODE_NONE) {
    in_dim = 1;
    out_dim = output_grad_domain.hi()[0] - output_grad_domain.lo()[0] + 1;
    effective_batch_size = output_grad_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  } else {
    in_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
    out_dim = output_grad_domain.hi()[0] - output_grad_domain.lo()[0] + 1;
    effective_batch_size = output_grad_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel<TI>(input_ptr, output_grad_ptr, kernel_grad_ptr,
      in_dim, out_dim, effective_batch_size,
      m->aggr, output_grad_domain.get_volume(), stream);

  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    //print_tensor<float>(output_grad_ptr, output_grad_domain.volume(), "[Embedding:backward:output_grad]");
    //print_tensor<float>(kernel_grad_ptr, kernel_grad_domain.get_volume(), "[Embedding:backward:weight_grad]");
    //print_tensor<TI>(input_ptr, input_domain.get_volume(), "[Embedding:backward:input]");
  }
}

__global__
void rand_generate_int64(int64_t* ptr, size_t size, int64_t p)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = i % p;
  }
}

bool Embedding::measure_operator_cost(Simulator* sim,
                                      const ParallelConfig& pc,
                                      CostMetrics& cost_metrics) const
{
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  sim->free_all();
  bool out_of_memory = false;
  int64_t *input_ptr = (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
  out_of_memory = out_of_memory || (input_ptr == NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  out_of_memory = out_of_memory || (output_ptr == NULL);
  float *weight_ptr = (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
  out_of_memory = out_of_memory || (weight_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }
  int in_dim = sub_input.dims[0].size;
  int out_dim = sub_input.dims[0].size;
  assert (sub_input.dims[1] == sub_output.dims[1]);
  int batch_size = sub_input.dims[1].size;

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Randomly initialize the intput tensor to avoid out of index range issues
  rand_generate_int64<<<GET_BLOCKS(sub_input.get_volume()), CUDA_NUM_THREADS, 0, stream>>>(
      input_ptr, sub_input.get_volume(), num_entries);
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(input_ptr, output_ptr, weight_ptr, in_dim, out_dim, batch_size, this->aggr, sub_output.get_volume(), stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *weight_grad_ptr = (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
    out_of_memory = out_of_memory || (weight_grad_ptr == NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (output_grad_ptr == NULL);
    int64_t *input_grad_ptr = (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
    out_of_memory = out_of_memory || (input_grad_ptr == NULL);
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, weight_grad_ptr, in_dim, out_dim, batch_size,
        this->aggr, sub_output.get_volume(), stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

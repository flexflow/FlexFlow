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

#include "flexflow/ops/dropout.h"
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
using Legion::Memory;
using Legion::Machine;

OpMeta* Dropout::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Dropout* dropout = (Dropout*) task->args;
  FFHandler handle = *((FFHandler*) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
      .only_kind(Memory::GPU_FB_MEM).best_affinity_to(task->target_proc).first();
  assert(input_domain == output_domain);
  DropoutMeta* m = new DropoutMeta(handle, dropout, gpu_mem, output_domain);
  return m;
}

void Dropout::forward_kernel(DropoutMeta *m,
                             float const *input_ptr,
                             float *output_ptr,
                             cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnDropoutForward(m->handle.dnn, m->dropoutDesc,
      m->inputTensor, input_ptr, m->outputTensor, output_ptr,
      m->reserveSpace, m->reserveSpaceSize));
}

__host__
void Dropout::forward_task(const Task* task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime* runtime)
{
  //float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Dropout* dropout = (const Dropout*) task->args;
  DropoutMeta* m = *((DropoutMeta**) task->local_args);
  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(m, input_ptr, output_ptr, stream);
}

void Dropout::backward_kernel(DropoutMeta *m,
                              float const *output_grad_ptr,
                              float *input_grad_ptr,
                              cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnDropoutBackward(m->handle.dnn, m->dropoutDesc,
      m->outputTensor, output_grad_ptr, m->inputTensor, input_grad_ptr,
      m->reserveSpace, m->reserveSpaceSize));
}

/*
  regions[0](I/O): input_grad
  regions[1](I): output_grad
*/
__host__
void Dropout::backward_task(const Task* task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime* runtime)
{
  //float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Dropout* dropout = (const Dropout*) task->args;
  DropoutMeta* m = *((DropoutMeta**) task->local_args);
  float* input_grad_ptr = helperGetTensorPointerRW<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(m, output_grad_ptr, input_grad_ptr, stream);
}

DropoutMeta::DropoutMeta(FFHandler handler,
                         const Dropout* dropout,
                         Memory gpu_mem,
                         const Domain& output_domain)
: OpMeta(handler)
{
  profiling = dropout->profiling;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  checkCUDNN(cudnnDropoutGetStatesSize(handle.dnn, &(dropoutStateSize)));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, output_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor, output_domain));
  checkCUDNN(cudnnDropoutGetReserveSpaceSize(outputTensor, &(reserveSpaceSize)));
  {
    // allocate memory for dropoutStates and reserveSpace
    size_t totalSize = dropoutStateSize + reserveSpaceSize;
    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
        Realm::Point<1, coord_t>(totalSize-1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(reserveInst, gpu_mem, bounds,
        field_sizes, 0, Realm::ProfilingRequestSet()).wait();
    dropoutStates = reserveInst.pointer_untyped(0, sizeof(char));
    reserveSpace = ((char*)dropoutStates) + dropoutStateSize;
  }
  //checkCUDA(cudaMalloc(&dropoutStates, dropoutStateSize));
  //checkCUDA(cudaMalloc(&reserveSpace, reserveSpaceSize));
  checkCUDNN(cudnnSetDropoutDescriptor(
    dropoutDesc, handle.dnn, dropout->rate, dropoutStates, dropoutStateSize, dropout->seed
  ));
}

DropoutMeta::~DropoutMeta(void)
{
  reserveInst.destroy();
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyDropoutDescriptor(dropoutDesc));
}

bool Dropout::measure_operator_cost(Simulator* sim,
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
  assert(sub_input.get_domain() == sub_output.get_domain());
  DropoutMeta *m = new DropoutMeta(sim->handler, this, sim->memory,
      sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert (input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, output_grad_ptr, input_grad_ptr, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Meausre Dropout] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Meausre Dropout] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }
  // Free dropoutmeta
  delete m;
  return true;
}

}; // namespace

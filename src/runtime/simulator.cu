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

#include "simulator.h"
#include "model.h"
#include "cuda_helper.h"
#include "ops/linear.h"
#include "ops/conv_2d.h"
#include "ops/pool_2d.h"
#include "ops/concat.h"
#include "ops/element_unary.h"

using namespace Legion;

typedef long long int coord_t;

typedef Realm::Point<1, coord_t> Point1;
typedef Realm::Rect<1, coord_t> Rect1;

Simulator::Simulator(const FFModel* model,
                     FFHandler _handler,
                     Memory _memory,
                     MachineModel *machine)
: memory(_memory), handler(_handler),
  offset(0), warmup_times(5), repeat_times(10),
  computationMode(model->config.computationMode)
{
  // Allocate simulator memory
  Rect1 bounds(Point1(0), Point1(0));
  std::vector<size_t> field_sizes;
  field_sizes.push_back(model->config.simulator_work_space_size);
  Realm::RegionInstance::create_instance(simulatorInst,
      memory, bounds, field_sizes, 0, Realm::ProfilingRequestSet()).wait();
  base_ptr = (char*)simulatorInst.pointer_untyped(0, sizeof(char));
  capacity = model->config.simulator_work_space_size;

  // Set cublas/cudnn streams to allow Realm catch the events
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(handler.blas, stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
#endif

  size_t max_num_tasks = 1024 * 1024;

  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);
  conv2d_meta = new Conv2DMeta(handler);
  linear_meta = new LinearMeta(handler, 4096);
  pool2d_meta = new Pool2DMeta(handler);
  ele_unary_meta = new ElementUnaryMeta(handler);
  ele_binary_meta = new ElementBinaryMeta(handler);
  //softmax_meta = new SoftmaxMeta(handler);
  batch_matmul_meta = new BatchMatmulMeta(handler);
  concat_meta = new ConcatMeta(handler);
  //dropout_meta = new DropoutMeta(handler);
  transpose_meta = new TransposeMeta(handler);
  this->machine = machine;
  segment_size = model->config.simulator_segment_size;
  max_num_segments = model->config.simulator_max_num_segments;
  // Initialize task manager
  task_manager = new TaskManager(max_num_tasks);
}

Simulator::~Simulator(void)
{
  simulatorInst.destroy();
}

__host__
void Simulator::strategy_search_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
  // This method should no longer be used
  assert(false);
  FFModel* model = *((FFModel**) task->args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
         .only_kind(Memory::GPU_FB_MEM).best_affinity_to(task->target_proc).first();
  MachineModel *machine;
  if (model->config.machine_model_version == 0) {
    machine = (MachineModel *) new SimpleMachineModel(model->config.numNodes, model->config.workersPerNode, gpu_mem.capacity());
  }
  else if (model->config.machine_model_version == 1 and !model->config.machine_model_file.empty()) {
    machine = (MachineModel *) new EnhancedMachineModel(model->config.machine_model_file, gpu_mem.capacity());
  }
  else {
    assert(false && "machine model creation error: currently only support machine-model-version = 0 or 1. When machine-model-version = 1, machine-model-file should not be empty.");
  }
  // Assume this task is running on GPU0
  Simulator* simulator = new Simulator(model, model->handlers[0], gpu_mem, machine);
  model->simulator = simulator;
  // Set cublas/cudnn streams to allow Realm catch the events
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(simulator->handler.blas, stream));
  checkCUDNN(cudnnSetStream(simulator->handler.dnn, stream));
#endif
  if (model->config.computationMode == COMP_MODE_TRAINING) {
    fprintf(stderr, "MCMC search configuration: budget(%zu) alpha(%.8lf) mode(TRAINING)\n",
        model->config.search_budget, model->config.search_alpha);
  } else {
    fprintf(stderr, "MCMC search configuration: budget(%zu) alpha(%.8lf) mode(INFERENCE)\n",
        model->config.search_budget, model->config.search_alpha);
  }
  // Start from data
  // memFBImpl->free_bytes_local(offset, model->config.simulator_work_space_size);
  delete(simulator);
  model->simulator = NULL;
  delete(machine);
}


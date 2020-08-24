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
#include "realm/runtime_impl.h"
#include "realm/cuda/cuda_module.h"

Simulator::Simulator(const FFModel* model,
                     FFHandler handler,
                     void* _base_ptr, size_t _capacity)
: base_ptr((char*)_base_ptr), capacity(_capacity), offset(0),
warmup_times(5), repeat_times(10)
{
  float inter_gpu_bandwidth = 20 * 1024 * 1024.0f; /* MB/ms*/
  float inter_node_bandwidth = 12 * 1024 * 1024.0f / model->config.numNodes; /* MB/ms*/
  float gpu_dram_bandwidth = 16 * 1024 * 1024.0f; /* MB/ms*/
  size_t max_num_tasks = 1024 * 1024;

  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);
  conv2d_meta = new Conv2DMeta(handler);
  linear_meta = new LinearMeta(handler, 4096);
  pool2d_meta = new Pool2DMeta(handler);
  int num_nodes = model->config.numNodes;
  int gpus_per_node = model->config.workersPerNode;
  total_num_devices = num_nodes * gpus_per_node;
  // Create GPU compute device
  for (int i = 0; i < num_nodes; i++) 
    for (int j = 0; j < gpus_per_node; j++) {
      id_to_compute_device[i*gpus_per_node+j] = new Device(Device::DEVICE_GPU,
          i, i*gpus_per_node+j);
    }
  // Create inter GPU comm devices:
  for (int i = 0; i < total_num_devices; i++)
    for (int j = 0; j < total_num_devices; j++) {
      Device* src = id_to_compute_device[i];
      Device* dst = id_to_compute_device[j];
      if (src->node_id == dst->node_id && src != dst) {
        int hash = i * total_num_devices + j;
        ids_to_inter_gpu_comm_device[hash] = new Device(Device::DEVICE_COMM,
            inter_gpu_bandwidth);
      }
    }
  // Create gpu<->dram comm devices
  for (int i = 0; i < total_num_devices; i++) {
    id_to_gputodram_comm_device[i] = new Device(Device::DEVICE_COMM,
        gpu_dram_bandwidth);
    id_to_dramtogpu_comm_device[i] = new Device(Device::DEVICE_COMM,
        gpu_dram_bandwidth);
  }
  // Create inter node comm devices
  for (int i = 0; i < num_nodes; i++)
    for (int j = 0; j < num_nodes; j++)
      if (i != j) {
        int hash = i * total_num_devices + j;
        ids_to_inter_node_comm_device[hash] = new Device(Device::DEVICE_COMM,
            inter_node_bandwidth);
      }
  // Initialize task manager
  task_manager = new TaskManager(max_num_tasks);
}

__host__
void Simulator::strategy_search_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
  const FFModel* model = *((FFModel**) task->args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
         .only_kind(Memory::GPU_FB_MEM).best_affinity_to(task->target_proc).first();
  Realm::MemoryImpl* memImpl =
      Realm::get_runtime()->get_memory_impl(gpu_mem);
  Realm::Cuda::GPUFBMemory* memFBImpl = (Realm::Cuda::GPUFBMemory*) memImpl;
  off_t offset = memFBImpl->alloc_bytes_local(model->config.simulator_work_space_size);
  void* base_ptr = memFBImpl->get_direct_ptr(offset, 0);
  // Assume this task is running on GPU0
  Simulator* simulator = new Simulator(model, model->handlers[0], base_ptr,
      model->config.simulator_work_space_size);
  std::map<Op*, ParallelConfig> strategies;
  model->optimize(simulator, strategies, model->config.search_budget, model->config.search_alpha);
  if (model->config.export_strategy_file.length() > 0) {
    std::map<Op*, ParallelConfig>::const_iterator iter;
    std::map<MappingTagID, ParallelConfig> strategy_output;
    for (iter = strategies.begin(); iter != strategies.end(); iter++) {
      strategy_output[FFConfig::get_hash_id(std::string(iter->first->name))] = iter->second;
    }
    save_strategies_to_file(model->config.export_strategy_file, strategy_output);
  }
  // Start from data
  memFBImpl->free_bytes_local(offset, model->config.simulator_work_space_size);
  delete(simulator);
}


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

Simulator::Simulator(FFModel* model, void* _base_ptr, size_t _capacity)
: handle(_handle), base_ptr((char*)_base_ptr), capacity(_capacity), offset(0),
warmup_times(5), repeat_times(10)
{
  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);
  conv2d_meta = new Conv2DMeta(model->config.handle);
  linear_meta = new LinearMeta(model->config.handle, 4096);
  int num_nodes = model->config.numNodes;
  int gpus_per_node = model->config.workersPerNode;
  total_num_devices = num_nodes * gpus_per_node;
  // Create GPU compute device
  for (int i = 0; i < num_nodes; i++) 
    for (int j = 0; j < gpus_per_node; j++) {
      id_to_comp_device[i*gpus_per_node+j] = new Device(DEVICE_COMPUTE, i, j);
    }
  // Create inter GPU comm devices:
  for (int i = 0; i < total_num_devices; i++)
    for (int j = 0; j < total_num_devices; j++) {
      Device* src = id_to_comp_device[i];
      Device* dst = id_to_comp_device[j];
      if (src->node_id == dst->node_id && src != dst) {
        int hash = i * total_num_devices + j;
        ids_to_inter_gpu_comm_device[hash] = new Device(DEVICE_COMM,
            SimulatorConfig::inter_gpu_bandwidth);
      }
    }
  // Create gpu<->dram comm devices
  for (int i = 0; i < total_num_devices; i++) {
    id_to_gputodram_comm_device[i] = new Device(DEVICE_COMM,
        SimulatorConfig::gpu_dram_bandwidth);
    id_to_dramtogpu_comm_device[i] = new Device(DEVICE_COMM,
        SimulatorConfig::gpu_dram_bandwidth);
  }
  // Create inter node comm devices
  for (int i = 0; i < num_nodes; i++)
    for (int j = 0; j < num_nodes; j++)
      if (i != j) {
        int hash = i * total_num_devices + j;
        ids_to_inter_node_comm_device[hash] = new Device(DEVICE_COMM,
            SimulatorConfig::inter_node_bandwidth);
      }
  // Initialize task manager
  task_manager = new TaskManager(SimulatorConfig::max_num_tasks);
}

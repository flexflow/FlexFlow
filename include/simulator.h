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
#ifndef _FLEXFLOW_SIMULATOR_H_
#define _FLEXFLOW_SIMULATOR_H_

#include "ffconst.h"
#include "config.h"

class Conv2DMeta;
class LinearMeta;
class Pool2DMeta;
class Op;
class FFModel;

class Device {
public:
  enum DeviceType {
    DEVICE_GPU,
    DEVICE_CPU,
    DEVICE_COMM,
  };
  Device(DeviceType type, int node_id, int gpu_id);
  Device(DeviceType type, float bandwidth);
public:
  int node_id, gpu_id;
  float bandwidth;
  DeviceType type;
};

class SimTask {
public:
  enum SimTaskType {
    TASK_FORWARD,
    TASK_BACKWARD,
    TASK_COMM,
    TASK_UPDATE,
    TASK_BARRIER,
  };
  SimTask();
  void add_next_task(SimTask* task);
public:
  float ready_time, run_time;
  SimTaskType type;
  Device* device;
  int counter;
  std::vector<SimTask*> next_tasks;
};

class SimTaskCompare {
public:
  bool operator() (SimTask* lhs, SimTask* rhs) {
    return lhs->ready_time > rhs->ready_time;
  }
};

class TaskManager {
public:
  TaskManager(size_t max_num_tasks);
  void reset();
  SimTask* new_barrier_task();
  SimTask* new_update_task();
  SimTask* new_comm_task();
  SimTask* new_forward_task(Op* op, int idx);
  SimTask* new_backward_task(Op* op, int idx);
  SimTask* get_forward_task(Op* op, int idx);
  SimTask* get_backward_task(Op* op, int idx);
private:
  SimTask* new_task();
public:
  size_t global_task_id, max_num_tasks;
  SimTask** tasks;
  std::map<size_t, SimTask*> hash_to_forward_task, hash_to_backward_task;
};

class Simulator {
public:
  Simulator(const FFModel* model, FFHandler handler, void* base_ptr, size_t capacity);
  void free_all();
  void* allocate(size_t num_elements, DataType type); 
  Device* get_compute_device_by_id(int device_id);
  Device* get_inter_gpu_comm_device_by_ids(int src_id, int dst_id);
  Device* get_inter_node_comm_device_by_ids(int src_id, int dst_id);
  Device* get_gpu_to_dram_comm_device_by_id(int gpu_id);
  Device* get_dram_to_gpu_comm_device_by_id(int gpu_id);
  void add_task_dependencies_with_xfer(
      SimTask* src_task, SimTask* dst_task, size_t intersect);
  float measure_op_forward_time(Op* op, const ParallelConfig& config);
  float measure_op_backward_time(Op* op, const ParallelConfig& config);
  float simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global);
  static void strategy_search_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime);
public:
  char* base_ptr;
  size_t capacity;
  off_t offset;
  int warmup_times, repeat_times;
  int total_num_devices;
  TaskManager* task_manager;
  cudaEvent_t start_event, end_event;
  std::map<int, Device*> id_to_compute_device;
  std::map<int, Device*> id_to_gputodram_comm_device;
  std::map<int, Device*> id_to_dramtogpu_comm_device;
  std::map<size_t, Device*> ids_to_inter_gpu_comm_device;
  std::map<size_t, Device*> ids_to_inter_node_comm_device;
  std::map<size_t, float> hash_to_op_forward_time;
  std::map<size_t, float> hash_to_op_backward_time;
public:
  Conv2DMeta* conv2d_meta;
  LinearMeta* linear_meta;
  Pool2DMeta* pool2d_meta;
};
#endif

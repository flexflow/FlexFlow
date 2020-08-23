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

Task::Task()
{}

TaskManager::TaskManager(size_t max_num_tasks)
{
  tasks = (Task**) malloc(sizeof(Task*) * max_num_tasks);
  for (size_t i = 0; i < max_num_tasks; i++) {
    tasks[i] = new Task();
  }
}

void TaskManager::reset()
{
  global_task_id = 0;
  hash_to_forward_task.clear();
  hash_to_backward_task.clear();
}

Task* TaskManager::new_task()
{
  Task* task = tasks[global_task_id++];
  task->nextTasks.clear();
  task->counter = 0;
  task->device = NULL;
  return task;
}

Task* TaskManager::new_comm_task()
{
  Task* task = new_task();
  task->type = TASK_COMM;
  return task; 
}

Task* TaskManager::new_forward_task(Op* op, int idx)
{
  Task* task = new_task();
  task->type = TASK_FORWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_forward_task[hash] = task;
}

Task* TaskManager::new_backward_task(Op* op, int idx)
{
  Task* task = new_task();
  task->type = TASK_BACKWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_backward_task[hash] = task;
}

Task* TaskManager::get_forward_task(Op* op, int idx)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_forward_task.find(hash) != hash_to_forward_task.end());
  return hash_to_forward_task[hash];
}

Task* TaskManager::get_backward_task(Op* op, int idx)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_backward_task.find(hash) != hash_to_backward_task.end());
  return hash_to_backward_task[hash];
}

void Simulator::free_all()
{
  offset = 0;
}

void* Simulator::allocate(size_t num_elements, DataType type)
{
  size_t element_size = 0;
  switch (type) {
    case DT_FLOAT:
      element_size = sizeof(float);
      break;
    case DT_DOUBLE:
      element_size = sizeof(double);
      break;
    case DT_INT32:
      element_size = sizeof(int32_t);
      break;
    case DT_INT64:
      element_size = sizeof(int64_t);
      break;
    case DT_BOOLEAN:
      element_size = sizeof(bool);
      break;
    default:
      assert(false);
  }
  void* ret_ptr = base_ptr + offset;
  offset += element_size * num_elements;
  return ret_ptr;
}

Device* Simulator::get_compute_device_by_id(int device_id)
{
  assert(id_to_compute_device.find(device_id) != id_to_compute_device.end());
  return id_to_compute_device[device_id];
}

Device* Simulator::get_inter_gpu_comm_device_by_ids(int src_id,
                                                    int dst_id)
{
  int hash = src_id * total_num_devices + dst_id;
  assert(ids_to_inter_gpu_comm_device.find(hash) != ids_to_inter_gpu_comm_device.end());
  return ids_to_inter_gpu_comm_device[hash];
}

Device* Simulator::get_gpu_to_dram_comm_device_by_id(int gpu_id)
{
  assert(id_to_gputodram_comm_device.find(gpu_id) != id_to_gputodram_comm_device.end());
  return id_to_gputodram_comm_device[gpu_id];
}

Device* Simulator::get_dram_to_gpu_comm_device_by_id(int gpu_id)
{
  assert(id_to_dramtogpu_comm_device.find(gpu_id) != id_to_dramtogpu_comm_device.end());
  return id_to_dramtogpu_comm_device[gpu_id];
}

Device* Simulator::get_inter_node_comm_device_by_ids(int src_id,
                                                     int dst_id)
{
  int hash = src_id * total_num_devices + dst_id;
  assert(ids_to_inter_node_comm_device.find(hash) != ids_to_inter_node_comm_device.end());
  return ids_to_inter_node_comm_device[hash];
}

void Simulator::add_task_dependencies_with_xfer(FFModel* model,
                                                Task* src_task,
                                                Task* dst_task,
                                                size_t intersect)
{
  if (src_task->device == dst_task->device) {
    src_task->add_next_task(dst_task);
  } else if (src_task->device->node_id == dst_task->device->node_id) {
    // Intra-node communication
    Task* task = task_manager->new_comm_task();
    task->device = get_inter_gpu_comm_device_by_ids(src_task->device->gpu_id,
                                                    dst_task->device->gpu_id);
    task->run_time = (float)intersect * sizeof(float) / task->device->bandwidth;
    src_task->add_next_task(task);
    task->add_next_task(dst_task);
  } else {
    // Inter-node communication
    Task* gpu_to_dram = task_manager->new_comm_task();
    gpu_to_dram->device = get_gpu_to_dram_comm_device_by_id(src_task->device->gpu_id);
    gpu_to_dram->run_time = (float)intersect * sizeof(float) / gpu_to_dram->device->bandwidth;
    Task* dram_to_dram = task_manager->new_comm_task();
    dram_to_dram->device = get_inter_node_comm_device_by_ids(src_task->device->node_id,
                                                             dst_task->device->node_id);
    dram_to_dram->run_time = (float)intersect * sizeof(float) / dram_to_dram->device->bandwidth;
    Task* dram_to_gpu = task_manager->new_comm_task();
    dram_to_gpu->device = get_dram_to_gpu_comm_device_by_id(dst_task->device->gpu_id);
    dram_to_gpu->run_time = (float)intersect * sizeof(float) / dram_to_gpu->device->bandwidth;
    src_task->add_next_task(gpu_to_dram);
    gpu_to_dram->add_next_task(dram_to_dram);
    dram_to_dram->add_next_task(dram_to_gpu);
    dram_to_gpu->add_next_task(dst_task);
  }
}

float Simulator::measure_op_forward_time(Op* op, const ParallelConfig& config)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(config.device_type);
  hash = hash * 31 + std::hash<int>()(config.nDims);
  for (int i = 0; i < config.nDims; i++)
    hash = hash * 31 + std::hash<int>()(config.dim[i]);
  if (hash_to_op_forward_time.find(hash) == hash_to_op_forward_time.end()) {
    float forward_time, backward_time;
    op->measure_compute_time(this, config, forward_time, backward_time);
    hash_to_op_forward_time[hash] = forward_time;
    // Check consistency betwek forward and backward
    assert(hash_to_op_backward_time.find(hash) == hash_to_op_backward_time.end());
    hash_to_op_backward_time[hash] = backward_time;
    return forward_time;
  } else {
    return hash_to_op_forward_time[hash];
  }
}

float Simulator::measure_op_backward_time(Op* op, const ParallelConfig& config)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(config.device_type);
  hash = hash * 31 + std::hash<int>()(config.nDims);
  for (int i = 0; i < config.nDims; i++)
    hash = hash * 31 + std::hash<int>()(config.dim[i]);
  if (hash_to_op_backward_time.find(hash) == hash_to_op_backward_time.end()) {
    float forward_time, backward_time;
    op->measure_compute_time(this, config, forward_time, backward_time);
    // Check consistency betwek forward and backward
    assert(hash_to_op_forward_time.find(hash) == hash_to_op_forward_time.end());
    hash_to_op_forward_time[hash] = forward_time;
    hash_to_op_backward_time[hash] = backward_time;
    return backward_time;
  } else {
    return hash_to_op_forward_time[hash];
  }
}

float Simulator::simulate_runtime(FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global)
{
  task_manager->reset();
  // Step 1: register forward and backward tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    float forward_time = measure_op_forward_time(op, config);
    float backward_time = measure_op_backward_time(op, config);
    for (size_t j = 0; j < config.num_parts(); j++) {
      Task* task1 = task_manager->new_forward_task(op, j);
      task1->device = get_compute_device_by_id(config.device_ids[j]);
      task1->run_time = forward_time;
      Task* task2 = task_manager->new_backward_task(op, j);
      task2->device = get_compute_device_by_id(config.device_ids[j]);
      task2->run_time = backward_time;
      task1->add_next_task(task2);
    }
  }
  // Step 2: insert dependencies and comm. tasks before compute tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    for (size_t j = 0; j < op->inputTensors.size(); j++) {
      Tensor t = op->inputTensors[j];
      Op* pre_op = t.owner_op;
      if (pre_op == NULL) continue;
      ParallelConfig pre_config = global.find(pre_op)->second;
      for (size_t dstId = 0; dstId < config.num_parts(); dstId ++) {
        Rect dstR = op->get_input_tensor_shape(config, j, dstId);
        for (size_t srcId = 0; srcId < pre_config.num_parts(); srcId ++) {
          Rect srcR = pre_op->get_output_tensor_shape(pre_config, t.owner_idx, srcId);
          if (intersect(srcR, dstR) > 0) {
            // Forward dependency
            {
              Task* dstT = task_manager->get_forward_task(op, dstId);
              Task* srcT = task_manager->get_forward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(model, srcT, dstT, intersect(srcR, dstR));
            }
            // Backward dependency
            {
              Task* dstT = task_manager->get_backward_task(op, dstId);
              Task* srcT = task_manager->get_backward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(model, dstT, srcT, intersect(srcR, dstR));
            }
          }
        }
      }
    }
  }
  // Step 3: add ready tasks into ready_queue
  std::set<Task*, TaskCompare> ready_queue;
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    for (size_t i = 0; i < config.num_parts(); i++) {
      Task* task = task_manager->get_forward_task(op, i);
      if (task->counter == 0)
        ready_queue.push(task);
    } 
  }
  // Step 4: perform simulation
  std::map<Device*, float> device_times;
  while (!ready_queue.empty()) {
    // Find the task with the earliest start time
    Task* t = *ready_queue.top();
    ready_queue.pop();
    float ready_time = 0;
    if (device_times.find(t->device) != device_times.end()) {
      ready_time = device_times[t->device];
    }
    float start_time = std::max(ready_time, t->ready_time);
    float end_time = start_time + t->run_time;
    device_times[t->device] = end_time;
    for (size_t i = 0; i < t->next_tasks.size(); i++) {
      Task* next = t->next_tasks[i];
      next->ready_time = std::max(next->ready_time, end_time);
      next->counter --;
      if (next->counter == 0) {
        ready_queue.push(next);
      }
    }
  }
}


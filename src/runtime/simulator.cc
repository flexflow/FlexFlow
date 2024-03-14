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

#include "flexflow/simulator.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"
#include "flexflow/parallel_ops/combine.h"
#include "flexflow/parallel_ops/partition.h"
#include "flexflow/parallel_ops/reduction.h"
#include "flexflow/parallel_ops/replicate.h"
#include "flexflow/utils/dot/dot_file.h"
#include "flexflow/utils/hash_utils.h"
#include "queue"
#include <memory>
#include <random>
#include <unordered_set>

namespace FlexFlow {

using namespace Legion;

LegionRuntime::Logger::Category log_sim("sim");
LegionRuntime::Logger::Category log_ps_sim("ps_sim");
LegionRuntime::Logger::Category log_xfer_sim("xfer_sim");
LegionRuntime::Logger::Category log_xfer_est("xfer_est");

// template class std::map<const Op*, ParallelConfig>; // for debugging in gdb
// template class std::map<const Op*, MachineView>; // for debugging in gdb

size_t CostMetrics::total_memory() const {
  return inputs_memory + outputs_memory + weights_memory;
}

float CostMetrics::total_memory_in_mb() const {
  float mem_mb = (float)((total_memory()) / 1e4) / 1e2;
  return mem_mb;
}

size_t CostMetrics::total_mem_diff_from(off_t sim_offset) const {
  return static_cast<size_t>(sim_offset) - total_memory();
}

int ParallelConfig::num_parts() const {
  int nparts = 1;
  for (int i = 0; i < nDims; i++) {
    nparts *= dim[i];
  }
  return nparts;
}

bool ParallelConfig::is_data_parallel() const {
  int nparts = 1;
  for (int i = 0; i < nDims; i++) {
    nparts *= dim[i];
    if ((i < nDims - 1) && (dim[i] > 1)) {
      return false;
    }
  }
  for (int i = 0; i < nparts; i++) {
    if (device_ids[i] != i) {
      return false;
    }
  }
  return true;
}

bool MachineResource::is_valid_machine_view(MachineView const &view) const {
  if (view.device_type == MachineView::GPU) {
    // Currently assume start_gpu_id == view.start_device_id
    assert(view.start_device_id == start_gpu_id);
    int last_device_id = start_gpu_id;
    for (int i = 0; i < view.ndims; i++) {
      last_device_id += (view.dim[i] - 1) * view.stride[i];
    }
    // Check that last device id in range
    if (last_device_id >= start_gpu_id + (num_nodes - 1) * all_gpus_per_node +
                              available_gpus_per_node) {
      return false;
    }
    // in case all_gpus_per_node > available_gpus_per_node
    if (all_gpus_per_node > available_gpus_per_node) {
      int used_gpus_per_node = 1;
      for (int i = 0; i < view.ndims; i++) {
        if (view.stride[i] < all_gpus_per_node) {
          used_gpus_per_node += (view.dim[i] - 1) * view.stride[i];
        }
      }
      if (used_gpus_per_node > available_gpus_per_node) {
        return false;
      }
    }
    return true;
  } else if (view.device_type == MachineView::CPU) {
    // Currently assume start_cpu_id == view.start_device_id
    assert(view.start_device_id == start_cpu_id);
    int last_device_id = start_cpu_id;
    for (int i = 0; i < view.ndims; i++) {
      last_device_id += (view.dim[i] - 1) * view.stride[i];
    }
    // Check that last device id in range
    if (last_device_id >= start_cpu_id + (num_nodes - 1) * all_cpus_per_node +
                              available_cpus_per_node) {
      return false;
    }
    // in case all_cpus_per_node > available_cpus_per_node
    if (all_cpus_per_node > available_cpus_per_node) {
      int used_cpus_per_node = 1;
      for (int i = 0; i < view.ndims; i++) {
        if (view.stride[i] < all_cpus_per_node) {
          used_cpus_per_node += (view.dim[i] - 1) * view.stride[i];
        }
      }
      if (used_cpus_per_node > available_cpus_per_node) {
        return false;
      }
    }
    return true;
  } else {
    assert(false && "Unsupported Device Type");
    return false;
  }
}

Device::Device(std::string const &name,
               DeviceType type,
               int node_id,
               int socket_id,
               int device_id)
    : name(name), type(type), node_id(node_id), socket_id(socket_id),
      device_id(device_id) {}

CompDevice::CompDevice(std::string const &name,
                       CompDevType comp_type,
                       int node_id,
                       int socket_id,
                       int device_id)
    : Device(name, Device::DEVICE_COMP, node_id, socket_id, device_id),
      comp_type(comp_type) {}

MemDevice::MemDevice(std::string const &name,
                     MemDevType mem_type,
                     int node_id,
                     int socket_id,
                     int device_id,
                     size_t capacity)
    : Device(name, Device::DEVICE_MEM, node_id, socket_id, device_id),
      mem_type(mem_type), capacity(capacity) {}

CommDevice::CommDevice(std::string const &name,
                       CommDevType comm_type,
                       int node_id,
                       int socket_id,
                       int device_id,
                       float latency,
                       float bandwidth)
    : Device(name, Device::DEVICE_COMM, node_id, socket_id, device_id),
      comm_type(comm_type), latency(latency), bandwidth(bandwidth) {}

static std::random_device rd;
static std::mt19937 gen = std::mt19937(rd());
static std::uniform_real_distribution<> std_uniform =
    std::uniform_real_distribution<>(0.0, 1.0);

NominalCommDevice::NominalCommDevice(std::string const &name,
                                     int device_id,
                                     int nnodes,
                                     NetworkRoutingStrategy *routing)
    : CommDevice(name, CommDevice::NW_NOMINAL, -1, -1, device_id, 0, 0),
      routing_strategy(routing), dirty(true), nnode(nnodes) {}

void NominalCommDevice::reset() {
  dirty = true;
  routes = {};
}

Route NominalCommDevice::expand_to_physical() const {
  if (dirty) {
    if (routing_strategy == nullptr) {
      assert("don't know how to route!" && false);
    }
    // std::cerr << name << " dirty... " << std::endl;
    *const_cast<EcmpRoutes *>(&routes) =
        routing_strategy->get_routes(device_id / nnode, device_id % nnode);
    *const_cast<bool *>(&dirty) = false;
  }

  assert(routes.first.size() > 0 || device_id / nnode == device_id % nnode);
  size_t pick = 0;
  double choice = std_uniform(gen);
  for (size_t i = 0; i < routes.first.size(); i++) {
    if (choice > routes.first[i]) {
      break;
    }
    pick = i;
  }
  Route ret = Route(routes.second[pick].begin(), routes.second[pick].end());
  return ret;
}

void NominalCommDevice::set_physical_paths(EcmpRoutes const &rs) {
  routes = rs;
  dirty = false;
}

EcmpRoutes const &NominalCommDevice::get_all_routes() {
  if (dirty) {
    if (routing_strategy == nullptr) {
      assert("don't know how to route!" && false);
    }
    // std::cerr << name << " dirty... " << std::endl;
    *const_cast<EcmpRoutes *>(&routes) =
        routing_strategy->get_routes(device_id / nnode, device_id % nnode);
    *const_cast<bool *>(&dirty) = false;
  }
  return routes;
}

SimTask::SimTask() {}

void SimTask::add_next_task(SimTask *task) {
  next_tasks.push_back(task);
  task->counter++;
}

std::string SimTask::get_type_str() const {
  switch (type) {
    case TASK_FORWARD:
      return "Forward";
    case TASK_BACKWARD:
      return "Backward";
    case TASK_COMM:
      return "Comm";
    case TASK_UPDATE:
      return "Update";
    case TASK_BARRIER:
      return "Barrier";
    default:
      assert(false && "Unknown task type");
  }
}

TaskManager::TaskManager(size_t _max_num_tasks)
    : max_num_tasks(_max_num_tasks) {
  tasks = (SimTask **)malloc(sizeof(SimTask *) * max_num_tasks);
  for (size_t i = 0; i < max_num_tasks; i++) {
    tasks[i] = new SimTask();
  }
}

void TaskManager::reset() {
  global_task_id = 0;
  hash_to_forward_task.clear();
  hash_to_backward_task.clear();
}

SimTask *TaskManager::new_task() {
  assert(global_task_id + 1 < max_num_tasks);
  SimTask *task = tasks[global_task_id++];
  task->ready_time = 0.0f;
  task->run_time = 0.0f;
  task->next_tasks.clear();
  task->counter = 0;
  task->device = NULL;
  task->mem = NULL;
  task->name.clear();

  task->xfer_size = 0;
  task->xfer_left = 0;
  task->store = true;

  return task;
}

SimTask *TaskManager::new_update_task() {
  SimTask *task = new_task();
  task->type = SimTask::TASK_UPDATE;
  return task;
}

SimTask *TaskManager::new_barrier_task() {
  SimTask *task = new_task();
  task->type = SimTask::TASK_BARRIER;
  return task;
}

SimTask *TaskManager::new_comm_task() {
  SimTask *task = new_task();
  task->type = SimTask::TASK_COMM;
  return task;
}

SimTask *TaskManager::new_comm_task(std::string const &name,
                                    CommDevice *comm_device,
                                    size_t message_size) {
  SimTask *task = new_task();
  task->type = SimTask::TASK_COMM;
  task->name = name;
  task->device = comm_device;
  task->run_time = comm_device->latency + message_size / comm_device->bandwidth;
  return task;
}

SimTask *TaskManager::new_forward_task(Op const *op, int idx) {
  SimTask *task = new_task();
  task->type = SimTask::TASK_FORWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_forward_task[hash] = task;
  task->name = op->name;
  return task;
}

SimTask *TaskManager::new_backward_task(Op const *op, int idx) {
  SimTask *task = new_task();
  task->type = SimTask::TASK_BACKWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_backward_task[hash] = task;
  task->name = op->name;
  return task;
}

SimTask *TaskManager::get_forward_task(Op const *op, int idx) {
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_forward_task.find(hash) != hash_to_forward_task.end());
  return hash_to_forward_task[hash];
}

SimTask *TaskManager::get_backward_task(Op const *op, int idx) {
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_backward_task.find(hash) != hash_to_backward_task.end());
  return hash_to_backward_task[hash];
}

void Simulator::free_all() {
  offset = 0;
}

void *Simulator::allocate(size_t num_elements, DataType type) {
  size_t element_size = data_type_size(type);
  void *ret_ptr = base_ptr + offset;
  offset += element_size * num_elements;
  if ((size_t)offset > capacity) {
    fprintf(stderr,
            "Simulator cannot measure some operators' performance."
            " Increate --simulator-workspace-size to at least %zd\n",
            offset);
    return NULL;
  }
  return ret_ptr;
}

void Simulator::add_task_dependencies_with_xfer(SimTask *src_task,
                                                SimTask *dst_task,
                                                size_t message_size,
                                                bool zero_cost) {
  std::vector<CommDevice *> path =
      machine->get_comm_path(src_task->mem, dst_task->mem);
  // print the communication path
  // printf("Message: %zu B\nPath from %s to %s is: ", message_size,
  // src_task->mem->name.c_str(), dst_task->mem->name.c_str()); for (size_t i =
  // 0; i < path.size(); i++) {
  //   printf("%s ", path[i]->name.c_str());
  // }
  // printf("\n");

  if (path.empty() || zero_cost) {
    log_xfer_sim.spew("Simulated xfer cost from %s to %s: 0ms",
                      src_task->name.c_str(),
                      dst_task->name.c_str());
    src_task->add_next_task(dst_task);
    return;
  }
  assert(message_size > 0);
  std::vector<std::vector<SimTask *>> all_tasks;
  // Limit the max number of segments per message
  int seg_size = segment_size;
  int num_segment = message_size / seg_size;
  if (message_size % seg_size != 0) {
    num_segment += 1;
  }
  if (num_segment > max_num_segments) {
    num_segment = max_num_segments;
    seg_size = message_size / num_segment;
  }
  // optional optimization: can reduce the simulation time, but could also
  // impact the accuracy of the simulation (a communication can be occupied by a
  // message for long time without be used by other concurrent communication
  //   if (path.size() == 1) {
  //     num_segment = 1;
  //     seg_size = message_size;
  //   }
  // Create all the comm tasks
  // Divide messages into segments
  for (size_t i = 0; i < path.size(); i++) {
    all_tasks.push_back({});
    for (int j = 0; j < num_segment; j++) {
      int cur_seg_size = seg_size;
      if (j == num_segment - 1) {
        cur_seg_size = message_size - (num_segment - 1) * seg_size;
      }
      std::string name = "seg " + std::to_string(j) + " from " +
                         src_task->name + " to " + dst_task->name;
      SimTask *cur_task =
          task_manager->new_comm_task(name, path[i], cur_seg_size);
      all_tasks[i].push_back(cur_task);
      if (j == 0) {
        log_xfer_sim.debug("Simulated xfer cost from %s to %s: %fms (%d)",
                           src_task->name.c_str(),
                           dst_task->name.c_str(),
                           cur_task->run_time,
                           cur_seg_size);
      }
    }
  }

  // Add dependencies among the comm tasks
  for (size_t i = 0; i < path.size(); i++) {
    for (int j = 0; j < num_segment; j++) {
      if (i == 0) {
        src_task->add_next_task(all_tasks[i][j]);
      }
      if (i == path.size() - 1) {
        all_tasks[i][j]->add_next_task(dst_task);
      }
      if (i > 0) {
        all_tasks[i - 1][j]->add_next_task(all_tasks[i][j]);
      }
    }
  }

  // Add special dependencies for upi_ins, upi_outs, nic_ins, and nic_outs to
  // prevent communication overlap between upi_ins and upi_outs, and between
  // nic_ins and nic_outs.
  if (num_segment > 1 and path.size() >= 2) {
    for (size_t i = 0; i < path.size(); i++) {
      for (int j = 0; j < num_segment - 1; j++) {
        if (((CommDevice *)all_tasks[i][j]->device)->comm_type ==
                CommDevice::NIC_OUT_COMM or
            ((CommDevice *)all_tasks[i][j]->device)->comm_type ==
                CommDevice::UPI_OUT_COMM) {
          all_tasks[i][j]->add_next_task(all_tasks[i - 1][j + 1]);
        }
      }
    }
  }
}

[[noreturn]] void handle_measure_operator_cost_unimplemented(Op const *op) {
  std::cerr << "measure_operator_cost not implemented for op " << op->name
            << " (type " << op->op_type << ")"
            << ". Please report this issue to the FlexFlow developers."
            << std::endl;
  std::abort();
}

CostMetrics Simulator::measure_operator_cost(Op const *op,
                                             ParallelConfig const &config) {
  assert(false);
#ifdef DEADCODE
  size_t hash = 17 * 31 + op->get_untyped_params_hash();
  hash = hash * 31 + std::hash<int>()(config.device_type);
  hash = hash * 31 + std::hash<int>()(config.nDims);
  for (int i = 0; i < config.nDims; i++) {
    hash = hash * 31 + std::hash<int>()(config.dim[i]);
  }
  std::unordered_map<size_t, CostMetrics>::const_iterator iter =
      hash_to_operator_cost.find(hash);
  if (iter == hash_to_operator_cost.end()) {
    CostMetrics cost_metrics;
    bool is_implemented = op->measure_operator_cost(this, config, cost_metrics);
    if (!is_implemented) {
      handle_measure_operator_cost_unimplemented(op);
    }
    hash_to_operator_cost[hash] = cost_metrics;
    return cost_metrics;
  } else {
    return iter->second;
  }
#endif
}

ParallelConfig Op::view_to_pc(MachineView const &view) const {
  ParallelConfig config;
  config.device_type = (ParallelConfig::DeviceType)view.device_type;
  const ParallelTensor output = this->outputs[0];
  config.nDims = output->num_dims;
  for (int i = 0; i < config.nDims; i++) {
    if (output->dims[i].parallel_idx == -1) {
      config.dim[i] = 1;
    } else {
      config.dim[i] = view.dim[output->dims[i].parallel_idx];
    }
  }

  std::vector<int> device_ids = view.device_ids();
  assert(device_ids.size() <= MAX_NUM_WORKERS);
  for (size_t i = 0; i < device_ids.size(); i++) {
    config.device_ids[i] = device_ids[i];
  }

  return config;
}

CostMetrics Simulator::measure_operator_cost(Op const *op,
                                             MachineView const &mv) {
  tl::optional<OperatorParameters> retrieved_params = get_op_parameters(op);
  if (retrieved_params.has_value()) {
    OperatorParameters params = retrieved_params.value();
    ProfilingRecordKey key{params, mv};
    if (this->strict_hash_to_operator_cost.find(key) ==
        this->strict_hash_to_operator_cost.end()) {
      CostMetrics cost_metrics{};
      bool is_implemented = op->measure_operator_cost(this, mv, cost_metrics);
      if (!is_implemented) {
        handle_measure_operator_cost_unimplemented(op);
      }
      op->estimate_sync_cost(this, mv, cost_metrics);
      this->strict_hash_to_operator_cost[key] = cost_metrics;
    }
    return this->strict_hash_to_operator_cost.at(key);
  }

  size_t hash = 17 * 31 + op->get_untyped_params_hash();
  hash = hash * 31 + std::hash<int>()(mv.device_type);
  hash = hash * 31 + std::hash<int>()(mv.ndims);
  for (int i = 0; i < mv.ndims; i++) {
    hash = hash * 31 + std::hash<int>()(mv.dim[i]);
  }
  std::unordered_map<size_t, CostMetrics>::const_iterator iter =
      hash_to_operator_cost.find(hash);

  if (iter == hash_to_operator_cost.end()) {
    CostMetrics cost_metrics{};
    bool is_implemented = op->measure_operator_cost(this, mv, cost_metrics);
    if (!is_implemented) {
      handle_measure_operator_cost_unimplemented(op);
    }
    op->estimate_sync_cost(this, mv, cost_metrics);
    hash_to_operator_cost[hash] = cost_metrics;
    return cost_metrics;
  } else {
    return iter->second;
  }
}

float Simulator::estimate_repartition_xfer_cost(
    int repartition_dim,
    int repartition_degree,
    ParallelTensorShape const &input_tensor_shape,
    ParallelTensorShape const &output_tensor_shape,
    MachineView const &source_view,
    MachineView const &sink_view) const {
  assert(source_view != sink_view);

  auto tensor_dim_to_mv_dim_mapping =
      output_tensor_shape.get_tensor_dim_to_mv_dim_mapping();
  size_t piece_size = input_tensor_shape.get_piece_size();
  piece_size /= repartition_degree;
  float max_xfer_cost = 0.0f;
  std::unordered_map<std::pair<int, int>, int> internode_transfers;
  for (Domain::DomainPointIterator it(sink_view.get_domain()); it; it++) {
    int sink_device = sink_view.get_device_id(*it);
    DomainPoint source_dp(*it);
    source_dp.point_data[tensor_dim_to_mv_dim_mapping.at(repartition_dim)] /=
        repartition_degree;
    int source_device = source_view.get_device_id(source_dp);

    float bandwidth = 0.0f;
    int src_node_id = machine->get_gpu(source_device)->node_id;
    int dst_node_id = machine->get_gpu(sink_device)->node_id;
    if (src_node_id == dst_node_id) {
      bandwidth = machine->get_intra_node_gpu_bandwidth();
      max_xfer_cost = std::max(max_xfer_cost, piece_size / bandwidth);
    } else {
      internode_transfers[{src_node_id, dst_node_id}] += piece_size;
    }
  }

  for (auto const &kv : internode_transfers) {
    max_xfer_cost = std::max(
        max_xfer_cost, kv.second / machine->get_inter_node_gpu_bandwidth());
  }

  return 2 * max_xfer_cost;
}

// estimate the data transfer costs from some op with view source_view to Op op
// with view sink_view
float Simulator::estimate_xfer_cost(Op const *op,
                                    int input_idx,
                                    MachineView const &source_view,
                                    MachineView const &sink_view) {
  // assert(tensor->is_valid_machine_view(source_view));
  // assert(tensor->is_valid_machine_view(sink_view));
  const ParallelTensor input_tensor = op->inputs[input_idx];
  if (input_tensor->owner_op->op_type == OP_INPUT) {
    return 0.0f;
  }

  if (op->is_parallel_op()) {
    assert(input_idx == 0);
    const ParallelTensor output_tensor = op->outputs[0];
    switch (op->op_type) {
      case OP_REPARTITION: {
        Repartition *rp = (Repartition *)op;
        return this->estimate_repartition_xfer_cost(rp->repartition_dim,
                                                    rp->repartition_degree,
                                                    input_tensor->get_shape(),
                                                    output_tensor->get_shape(),
                                                    source_view,
                                                    sink_view);
      }
      case OP_COMBINE: {
        Combine *combine = (Combine *)op;
        const ParallelTensor output_tensor = op->outputs[0];
        return this->estimate_repartition_xfer_cost(combine->combine_dim,
                                                    combine->combine_degree,
                                                    output_tensor->get_shape(),
                                                    input_tensor->get_shape(),
                                                    sink_view,
                                                    source_view);
      }
      case OP_REPLICATE: {
        Replicate *replicate = (Replicate *)op;
        ParallelTensorShape fake_input_shape = input_tensor->get_shape();
        fake_input_shape.dims[replicate->replicate_dim].size *=
            replicate->replicate_degree;
        return this->estimate_repartition_xfer_cost(replicate->replicate_dim,
                                                    replicate->replicate_degree,
                                                    fake_input_shape,
                                                    output_tensor->get_shape(),
                                                    source_view,
                                                    sink_view);
      }
      case OP_REDUCTION: {
        Reduction *reduction = (Reduction *)op;
        const ParallelTensor output_tensor = op->outputs[0];
        ParallelTensorShape fake_output_shape = output_tensor->get_shape();
        fake_output_shape.dims[reduction->reduction_dim].size *=
            reduction->reduction_degree;
        return this->estimate_repartition_xfer_cost(reduction->reduction_dim,
                                                    reduction->reduction_degree,
                                                    fake_output_shape,
                                                    input_tensor->get_shape(),
                                                    sink_view,
                                                    source_view);
      }
      case OP_FUSED_PARALLEL: {
        FusedParallelOp const *fused = (FusedParallelOp const *)op;
        const ParallelTensor input_tensor = op->inputs[0];
        const ParallelTensor output_tensor = op->outputs[0];
        ParallelTensorShape input_shape = input_tensor->get_shape();
        ParallelTensorShape output_shape = output_tensor->get_shape();
        // FIXME: we currently calculate an over estimation
        size_t input_piece_size = input_shape.get_piece_size();
        size_t output_piece_size = output_shape.get_piece_size();
        bool inter_node = false;
        for (Domain::DomainPointIterator it1(source_view.get_domain()); it1;
             it1++) {
          DomainPoint source_dp(*it1);
          int source_node_id = source_view.get_device_id(source_dp);
          for (Domain::DomainPointIterator it2(sink_view.get_domain()); it2;
               it2++) {
            DomainPoint sink_dp(*it2);
            int sink_node_id = sink_view.get_device_id(sink_dp);
            if (sink_node_id != source_node_id) {
              inter_node = true;
              break;
            }
          }
          if (inter_node) {
            break;
          }
        }
        float max_xfer_cost = 0.0f;
        if (inter_node) {
          // inter_node case
          float bandwidth = machine->get_inter_node_gpu_bandwidth();
          max_xfer_cost =
              std::max(input_piece_size, output_piece_size) / bandwidth;
        } else {
          // intra_node case
          float bandwidth = machine->get_intra_node_gpu_bandwidth();
          max_xfer_cost =
              std::max(input_piece_size, output_piece_size) / bandwidth;
        }
        return 2 * max_xfer_cost;
      }
      default:
        assert(false);
    }
  } else {
    // No cost if source_view == sink_view
    if (source_view == sink_view) {
      return 0.0f;
    }
    assert(source_view.ndims == sink_view.ndims);
    Domain d;
    d.dim = source_view.ndims;
    for (int i = 0; i < d.dim; i++) {
      assert(source_view.dim[i] == sink_view.dim[i]);
      d.rect_data[i] = 0;
      d.rect_data[i + d.dim] = source_view.dim[i] - 1;
    }
    const ParallelTensor input_tensor = op->inputs[input_idx];
    size_t total_size = data_type_size(input_tensor->data_type);
    for (int i = 0; i < input_tensor->num_dims; i++) {
      total_size *= input_tensor->dims[i].size / input_tensor->dims[i].degree;
    }
    float max_xfer_cost = 0.0f;
    for (Domain::DomainPointIterator it(d); it; it++) {
      int source_device = source_view.get_device_id(*it);
      int sink_device = sink_view.get_device_id(*it);
      float bandwidth = 0.0f;
      if (machine->get_gpu(source_device)->node_id ==
          machine->get_gpu(sink_device)->node_id) {
        bandwidth = machine->get_intra_node_gpu_bandwidth();
      } else {
        bandwidth = machine->get_inter_node_gpu_bandwidth();
      }
      max_xfer_cost = std::max(max_xfer_cost, 2 * total_size / bandwidth);
    }
    return max_xfer_cost;
  }
}

bool Op::estimate_sync_cost(Simulator *sim,
                            MachineView const &view,
                            CostMetrics &cost_metrics) const {
  // By default we assume an operator does not have sync cost
  // Implement a derived method for operators with parameters
  return true;
}

float Simulator::default_estimate_sync_cost(
    const ParallelDim tensor_dims[MAX_TENSOR_DIM],
    int tensor_ndims,
    MachineView const &view) {
  ParallelTensorShape tensor_shape(tensor_ndims, tensor_dims, DT_FLOAT);

  return this->default_estimate_sync_cost(
      tensor_shape, view, tensor_shape.get_num_replica_dims());
}

float Simulator::default_estimate_sync_cost(const ParallelTensor tensor,
                                            MachineView const &view,
                                            int num_replica_dims) {
  return this->default_estimate_sync_cost(
      tensor->get_shape(), view, num_replica_dims);
}

float Simulator::default_estimate_sync_cost(
    ParallelTensorShape const &tensor_shape,
    MachineView const &view,
    int num_replicate_dims) {
  // Currently only support 1 replicate_dim
  int num_replicas = tensor_shape.get_num_replicas();
  if (num_replicas == 1) {
    // No replications
    return 0.0f;
  } else {
    bool inter_node_sync = false;
    tl::optional<int> node = tl::nullopt;
    for (Domain::DomainPointIterator it(view.get_domain()); it; it++) {
      int my_device = view.get_device_id(*it);
      int my_node = machine->get_gpu(my_device)->node_id;
      if (node == tl::nullopt) {
        node = my_node;
      }
      if (my_node != node.value()) {
        inter_node_sync = true;
        break;
      }
    }
    float bandwidth = inter_node_sync
                          ? this->machine->get_inter_node_gpu_bandwidth()
                          : this->machine->get_intra_node_gpu_bandwidth();
    return 2 * tensor_shape.get_piece_size() / bandwidth;
  }
}

float Simulator::simulate_runtime(
    FFModel const *model,
    std::map<Op const *, ParallelConfig> const &global,
    CompMode comp_mode) {
  return this->simulate_runtime(model, global, comp_mode, "");
}

float Simulator::simulate_runtime(
    FFModel const *model,
    std::map<Op const *, ParallelConfig> const &global,
    CompMode comp_mode,
    std::string const &export_file_name) {
  // printf("%s\n", machine->to_string().c_str());
  task_manager->reset();
  // Step 1: register forward and backward tasks
  for (Op *op : model->operators) {
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    float forward_time = cost_metrics.forward_time;
    float backward_time = cost_metrics.backward_time;
    for (int j = 0; j < config.num_parts(); j++) {
      SimTask *task1 = task_manager->new_forward_task(op, j);
      task1->device = machine->get_gpu(config.device_ids[j]);
      task1->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
      task1->run_time = forward_time;
      if (comp_mode == COMP_MODE_TRAINING) {
        SimTask *task2 = task_manager->new_backward_task(op, j);
        task2->device = machine->get_gpu(config.device_ids[j]);
        task2->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
        task2->run_time = backward_time;
        task1->add_next_task(task2);
      }
    }
  }
  // Step 2: insert dependencies and comm. tasks before compute tasks
  for (Op *op : model->operators) {
    ParallelConfig config = global.find(op)->second;
    for (int j = 0; j < op->numInputs; j++) {
      ParallelTensor t = op->inputs[j];
      Op const *pre_op = t->owner_op;
      if (pre_op == NULL) {
        continue;
      }
      ParallelConfig pre_config = global.find(pre_op)->second;
      size_t element_size = data_type_size(t->data_type);
      for (int dstId = 0; dstId < config.num_parts(); dstId++) {
        Domain dstR = op->get_input_tensor_shape(config, j, dstId);
        for (int srcId = 0; srcId < pre_config.num_parts(); srcId++) {
          Domain srcR =
              pre_op->get_output_tensor_shape(pre_config, t->owner_idx, srcId);
          bool force_zero_cost = pre_op->op_type == OP_INPUT;
          if (dstR.intersection(srcR).get_volume() > 0) {
            // Forward dependency
            {
              SimTask *dstT = task_manager->get_forward_task(op, dstId);
              SimTask *srcT = task_manager->get_forward_task(pre_op, srcId);
              size_t xfer_size =
                  dstR.intersection(srcR).get_volume() * element_size;
              if (dstId == 0 && srcId == 0) {
                log_sim.debug("fwd xfer from %s to %s: %zu",
                              srcT->name.c_str(),
                              dstT->name.c_str(),
                              xfer_size);
              }
              add_task_dependencies_with_xfer(
                  srcT, dstT, xfer_size, force_zero_cost);
              // add_task_dependencies_with_xfer(srcT, dstT,
              // dstR.intersection(srcR).get_volume() * element_size);
            }
            // Backward dependency
            if (comp_mode == COMP_MODE_TRAINING) {
              SimTask *dstT = task_manager->get_backward_task(op, dstId);
              SimTask *srcT = task_manager->get_backward_task(pre_op, srcId);
              size_t xfer_size =
                  dstR.intersection(srcR).get_volume() * element_size;
              if (dstId == 0 && srcId == 0) {
                log_sim.debug("bwd xfer from %s to %s: %zu",
                              dstT->name.c_str(),
                              srcT->name.c_str(),
                              xfer_size);
              }
              add_task_dependencies_with_xfer(
                  dstT, srcT, xfer_size, force_zero_cost);
              // add_task_dependencies_with_xfer(dstT, srcT,
              // dstR.intersection(srcR).get_volume() * element_size);
            }
          }
        }
      }
    }
  }
#ifdef FF_USE_NCCL
  // Do nothing since we will calculate NCCL cost at the end
#else
  // Step 2.5: add finals tasks for each compute device to capture the returning
  // comm tasks from parameter servers
  std::vector<SimTask *> finals;
  for (int d = 0; d < machine->get_num_gpus(); d++) {
    SimTask *t = task_manager->new_barrier_task();
    t->device = machine->get_gpu(d);
    t->mem = machine->get_gpu_fb_mem(d);
    t->run_time = 0;
    finals.push_back(t);
  }

  if (model->config.search_overlap_backward_update &&
      comp_mode == COMP_MODE_TRAINING) {
    // Step 3a: consider backpropagation and weight update are overlapped
    for (int l = model->operators.size() - 1; l >= 0; l--) {
      Op *op = model->operators[l];
      size_t element_size =
          data_type_size(DT_FLOAT); // assume all weights have float elements
      ParallelConfig pc = global.find(op)->second;
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++) {
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            // Add a compute task for parameter update
            SimTask *updateT = task_manager->new_update_task();
            updateT->device = machine->get_gpu(pc.device_ids[firstId]);
            updateT->mem = machine->get_gpu_fb_mem(pc.device_ids[firstId]);
            // TODO add parameter synchronization time
            updateT->run_time = 0.0f; // Assume update task takes no time
            for (int nextId = firstId + 1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                // Add comm. tasks from backT to updateT
                SimTask *backT = task_manager->get_backward_task(op, nextId);
                add_task_dependencies_with_xfer(
                    backT, updateT, firstR.get_volume() * element_size);
                // Add comm. tasks from updateT to finalT
                SimTask *finalT = finals[backT->device->device_id];
                add_task_dependencies_with_xfer(
                    updateT, finalT, firstR.get_volume() * element_size);
              }
            }
          }
        }
      }
    }
  } else if (comp_mode == COMP_MODE_TRAINING) {
    // Step 3b: Bulk Synchronous Model
    // Add a per-device barrier before weight update
    std::vector<SimTask *> barriers;
    for (int d = 0; d < machine->get_num_gpus(); d++) {
      SimTask *t = task_manager->new_barrier_task();
      t->device = machine->get_gpu(d);
      t->mem = machine->get_gpu_fb_mem(d);
      t->run_time = 0;
      barriers.push_back(t);
    }
    for (size_t l = 0; l < model->operators.size(); l++) {
      Op *op = model->operators[l];
      ParallelConfig pc = global.find(op)->second;
      for (int j = 0; j < pc.num_parts(); j++) {
        SimTask *backT = task_manager->get_backward_task(op, j);
        backT->add_next_task(barriers[backT->device->device_id]);
      }
    }
    for (size_t l = 0; l < model->operators.size(); l++) {
      Op *op = model->operators[l];
      ParallelConfig pc = global.find(op)->second;
      size_t element_size =
          data_type_size(DT_FLOAT); // assume all weights have float elements
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++) {
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            // Add a compute task for parameter update
            SimTask *updateT = task_manager->new_update_task();
            updateT->device = machine->get_gpu(pc.device_ids[firstId]);
            updateT->mem = machine->get_gpu_fb_mem(pc.device_ids[firstId]);
            updateT->run_time = 0.0f; // Assume update task takes no time
            barriers[updateT->device->device_id]->add_next_task(updateT);
            for (int nextId = firstId + 1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                SimTask *backT = task_manager->get_backward_task(op, nextId);
                assert(backT->device->device_id == pc.device_ids[nextId]);
                SimTask *barrierT = barriers[backT->device->device_id];
                // Add comm. tasks from barrierT to updateT
                add_task_dependencies_with_xfer(
                    barrierT, updateT, firstR.get_volume() * element_size);
                // Add comm. tasks from updateT to finalT
                SimTask *finalT = finals[backT->device->device_id];
                add_task_dependencies_with_xfer(
                    updateT, finalT, firstR.get_volume() * element_size);
              }
            }
          }
        }
      }
    }
  } else {
    assert(comp_mode == COMP_MODE_INFERENCE);
  }
#endif
  // Step 4: add ready tasks into ready_queue
  std::priority_queue<SimTask *, std::vector<SimTask *>, SimTaskCompare>
      ready_queue;
  for (size_t i = 0; i < task_manager->global_task_id; i++) {
    if (task_manager->tasks[i]->counter == 0) {
      ready_queue.push(task_manager->tasks[i]);
    }
  }
  // Step 5: perform simulation
  float sim_time = 0.0f;
  std::map<Device *, float> device_times;
  size_t idx = 0;
  DotFile<SimTask *> taskGraph;
  bool export_taskgraph = (export_file_name != "");
  if (export_taskgraph) {
    taskGraph.set_filename(export_file_name);
  }
  while (!ready_queue.empty()) {
    // Find the task with the earliest start time
    SimTask *cur_task = ready_queue.top();
    ready_queue.pop();
    float ready_time = 0;
    if (device_times.find(cur_task->device) != device_times.end()) {
      ready_time = device_times[cur_task->device];
    }
    float start_time = std::max(ready_time, cur_task->ready_time);
    float end_time = start_time + cur_task->run_time;
    device_times[cur_task->device] = end_time;
    if (export_taskgraph) {
      std::map<std::string, std::string> nodeAttrs;
      std::ostringstream label;
      label << "\"{ ";
      if (!(cur_task->name).empty()) {
        label << cur_task->name << " | ";
      }
      label << cur_task->get_type_str() << " | ";
      label << "{ " << start_time << " | " << end_time << " }";
      label << " }\"";
      nodeAttrs["label"] = label.str();
      nodeAttrs["shape"] = "record";
      taskGraph.add_node(cur_task, nodeAttrs);
    }
    // printf("task[%lu] type(%d) run_time(%.4lf) ready_time(%.4lf)
    // start_time(%.4lf) device(%s)\n",
    //       idx, cur_task->type, cur_task->run_time, ready_time, start_time,
    //       (cur_task->device->name).c_str());
    if (end_time > sim_time) {
      sim_time = end_time;
    }
    for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
      SimTask *next = cur_task->next_tasks[i];
      if (export_taskgraph) {
        taskGraph.add_edge(cur_task, next);
      }
      next->ready_time = std::max(next->ready_time, end_time);
      next->counter--;
      if (next->counter == 0) {
        ready_queue.push(next);
      }
    }
    idx++;
  }
  if (export_taskgraph) {
    taskGraph.close();
  }
  // Assert all tasks were processed
  assert(idx == task_manager->global_task_id);
#ifdef FF_USE_NCCL
  if (comp_mode == COMP_MODE_TRAINING) {
    std::unordered_set<Op const *> possible_syncs(model->operators.begin(),
                                                  model->operators.end());
    std::unordered_map<Op const *, std::unique_ptr<OpSyncTask>> tasks;
    assert(std::numeric_limits<float>::has_quiet_NaN);
    for (Op const *op : model->operators) {
      tasks[op] = std::unique_ptr<OpSyncTask>(
          new OpSyncTask{op, 0, std::numeric_limits<float>::quiet_NaN()});
    }
    for (Op const *op : model->operators) {
      for (int i = 0; i < op->numInputs; i++) {
        Op const *src = op->inputs[i]->owner_op;
        possible_syncs.erase(src);

        tasks[src]->unsatisfied_dependencies++;
      }
    }
    assert(possible_syncs.size() == 1);

    std::vector<bool> available_devices(this->machine->get_num_gpus(), true);

    std::priority_queue<OpSyncTask *,
                        std::vector<OpSyncTask *>,
                        OpSyncTaskEarliestFirst>
        sync_ready_queue;

    float sync_sim_time = 0.0f;

    size_t syncs_processed = 0;
    while (possible_syncs.size() > 0 || !sync_ready_queue.empty()) {
      Op const *to_run = nullptr;
      for (Op const *op : possible_syncs) {
        bool can_be_run = true;
        ParallelConfig config = global.find(op)->second;
        for (int j = 0; j < config.num_parts(); j++) {
          can_be_run &= available_devices[config.device_ids[j]];
        }
        if (can_be_run) {
          to_run = op;
          break;
        }
      }
      if (to_run != nullptr) {
        possible_syncs.erase(to_run);
        float sync_run_time = 0.0f;
        OpSyncTask *task = tasks.at(to_run).get();
        Op const *op = to_run;
        ParallelConfig pc = global.find(op)->second;
        size_t element_size =
            data_type_size(DT_FLOAT); // assume all weights have float elements

        for (int j = 0; j < pc.num_parts(); j++) {
          available_devices[pc.device_ids[j]] = false;
        }

        for (int j = 0; j < op->numWeights; j++) {
          std::set<int> synched;
          for (int firstId = 0; firstId < pc.num_parts(); firstId++) {
            if (synched.find(firstId) == synched.end()) {
              synched.insert(firstId);
              Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
              Device *firstDevice = machine->get_gpu(pc.device_ids[firstId]);
              float nccl_time = 0.0f;
              for (int nextId = firstId + 1; nextId < pc.num_parts();
                   nextId++) {
                Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
                if (firstR.intersection(nextR).get_volume() > 0) {
                  // Assert all or nothing:
                  // The two weights must be fully overlapped or not at all
                  assert(firstR == nextR);
                  assert(synched.find(nextId) == synched.end());
                  synched.insert(nextId);
                  Device *nextDevice = machine->get_gpu(pc.device_ids[nextId]);
                  // Compute the bandwidth between firstDevice/nextDevice
                  float bandwidth = 0.0f;
                  if (firstDevice->node_id == nextDevice->node_id) {
                    bandwidth = machine->get_intra_node_gpu_bandwidth();
                  } else {
                    bandwidth = machine->get_inter_node_gpu_bandwidth();
                  }
                  // printf("[NCCL Time] Op(%s) Weight(%d) firstId(%d)
                  // nextId(%d): volume is %f\n", op->name, j, firstId, nextId,
                  // (float)firstR.get_volume());
                  nccl_time = std::max(nccl_time,
                                       2 * (float)firstR.get_volume() *
                                           element_size / bandwidth);
                }
              }
              sync_run_time += nccl_time;
            }
          }
        }

        task->finish_time = sync_sim_time + sync_run_time;
        sync_ready_queue.push(task);
        log_ps_sim.debug("Push sync task for %s\n", task->op->name);
        log_ps_sim.debug("  Time: %fms\n", sync_sim_time);
      } else {
        OpSyncTask *completed = sync_ready_queue.top();
        sync_ready_queue.pop();
        syncs_processed++;
        sync_sim_time = completed->finish_time;
        log_ps_sim.debug("Pop sync task for %s", completed->op->name);
        log_ps_sim.debug("  Time: %fms", sync_sim_time);
        ParallelConfig config = global.find(completed->op)->second;
        for (int j = 0; j < config.num_parts(); j++) {
          assert(!available_devices[config.device_ids[j]]);
          available_devices[config.device_ids[j]] = true;
        }
        for (int i = 0; i < completed->op->numInputs; i++) {
          OpSyncTask *dependent_task =
              tasks.at(completed->op->inputs[i]->owner_op).get();
          assert(dependent_task->unsatisfied_dependencies > 0);
          assert(std::isnan(dependent_task->finish_time));
          dependent_task->unsatisfied_dependencies--;
          if (dependent_task->unsatisfied_dependencies == 0) {
            possible_syncs.insert(dependent_task->op);
          }
        }
      }
    }
    assert(syncs_processed == model->operators.size());
    log_ps_sim.debug("Sync sim time: %fms", sync_sim_time);
    sim_time += sync_sim_time;
  } else {
    assert(comp_mode == COMP_MODE_INFERENCE);
  }
#endif
  // Step 6: add penalty to strategies that exceed the memory limits on devices
  std::vector<size_t> gpu_mem_usage(machine->get_num_gpus(), 0);
  float memory_penalty = 0.0f;
  for (size_t l = 0; l < model->operators.size(); l++) {
    Op *op = model->operators[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    size_t memory_requirement = cost_metrics.total_memory();
    for (int j = 0; j < config.num_parts(); j++) {
      gpu_mem_usage[config.device_ids[j]] += memory_requirement;
    }
  }
  if (export_file_name != "") {
    for (int i = 0; i < machine->get_num_gpus(); i++) {
      printf("Before penalty, dev id %d, usage %zu \n", i, gpu_mem_usage[i]);
    }
  }
  // Penalize the total runtiem by 1ms if we exceed the memory budget by 1MB
  for (int i = 0; i < machine->get_num_gpus(); i++) {
    MemDevice *gpu_fb_mem = machine->get_gpu_fb_mem(i);
    if (gpu_mem_usage[i] > gpu_fb_mem->capacity and gpu_fb_mem->capacity >= 0) {
      memory_penalty += (gpu_mem_usage[i] - gpu_fb_mem->capacity) * 1e-6;
    }
  }
  // if (memory_penalty > 0.0f)
  //   printf("Memory penalty = %.4lf ms\n", memory_penalty);
  return sim_time + memory_penalty;
}

float LogicalTaskgraphBasedSimulator::simulate_runtime(
    FFModel const *model,
    std::map<Op const *, ParallelConfig> const &global,
    CompMode comp_mode,
    std::string const &export_file_name) {
#ifdef WRITE_NETWORK_TRANSFER
  network_transfer_log.open("network.log");
#endif
  // printf("%s\n", machine->to_string().c_str());
  task_manager->reset();
  std::unordered_map<SimTask *, Op *> task_to_op;
  // Step 1: register forward and backward tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op *op = model->operators[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    float forward_time = cost_metrics.forward_time;
    float backward_time = cost_metrics.backward_time;
    // SimTask *ar_task = nullptr;
    for (int j = 0; j < config.num_parts(); j++) {
      SimTask *task1 = task_manager->new_forward_task(op, j);
      task_to_op[task1] = op;
      task1->device = machine->get_gpu(config.device_ids[j]);
      task1->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
      task1->run_time = forward_time;

      if (comp_mode == COMP_MODE_TRAINING) {
        SimTask *task2 = task_manager->new_backward_task(op, j);
        task_to_op[task2] = op;
        task2->device = machine->get_gpu(config.device_ids[j]);
        task2->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
        task2->run_time = backward_time;
        task1->add_next_task(task2);
      }
    }
  }

  for (size_t l = 0; l < model->layers.size(); l++) {
    Op *op = model->operators[l];
    ParallelConfig config = global.find(op)->second;
    size_t element_size = data_type_size(DT_FLOAT);
    // NER step: add allreduce task after backward propogation
    for (int j = 0; j < op->numWeights; j++) {
      std::set<int> synched;
      std::vector<int> node_ids;
      for (int firstId = 0; firstId < config.num_parts(); firstId++) {
        if (synched.find(firstId) == synched.end()) {
          synched.insert(firstId);
          Domain firstR = op->get_weight_tensor_shape(config, j, firstId);
          size_t xfer_size = firstR.get_volume() * element_size;
          node_ids.push_back(config.device_ids[firstId]);
          for (int nextId = firstId + 1; nextId < config.num_parts();
               nextId++) {
            Domain nextR = op->get_weight_tensor_shape(config, j, nextId);
            if (firstR.intersection(nextR).get_volume() > 0) {
              // Assert all or nothing:
              // The two weights must be fully overlapped or not at all
              assert(firstR == nextR);
              assert(synched.find(nextId) == synched.end());
              synched.insert(nextId);
              node_ids.push_back(config.device_ids[nextId]);
            }
          }

          SimTask *ar_task =
              task_manager->new_allreduce_task(op, node_ids, xfer_size);
          task_to_op[ar_task] = op;

          for (int dstId = 0; dstId < config.num_parts(); dstId++) {
            task_manager->get_backward_task(op, dstId)->add_next_task(ar_task);
          }
        }
      }
    }
  }

  // Step 2: insert dependencies and comm. tasks before compute tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op *op = model->operators[l];
    ParallelConfig config = global.find(op)->second;
    for (int j = 0; j < op->numInputs; j++) {
      ParallelTensor t = op->inputs[j];
      Op const *pre_op = t->owner_op;
      if (pre_op == NULL) {
        continue;
      }
      ParallelConfig pre_config = global.find(pre_op)->second;
      size_t element_size = data_type_size(t->data_type);
      for (int dstId = 0; dstId < config.num_parts(); dstId++) {
        Domain dstR = op->get_input_tensor_shape(config, j, dstId);
        for (int srcId = 0; srcId < pre_config.num_parts(); srcId++) {
          Domain srcR =
              pre_op->get_output_tensor_shape(pre_config, t->owner_idx, srcId);
          if (dstR.intersection(srcR).get_volume() > 0) {
            // Forward dependency
            {
              SimTask *dstT = task_manager->get_forward_task(op, dstId);
              SimTask *srcT = task_manager->get_forward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(
                  srcT,
                  dstT,
                  dstR.intersection(srcR).get_volume() * element_size);
            }
            // Backward dependency
            if (comp_mode == COMP_MODE_TRAINING) {
              SimTask *dstT = task_manager->get_backward_task(op, dstId);
              SimTask *srcT = task_manager->get_backward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(
                  dstT,
                  srcT,
                  dstR.intersection(srcR).get_volume() * element_size);
            }
          }
        }
      }
    }
  }

  // Step 4: add ready tasks into ready_queue
  std::priority_queue<SimTask *, std::vector<SimTask *>, SimTaskCompare>
      ready_queue;
  for (size_t i = 0; i < task_manager->global_task_id; i++) {
    if (task_manager->tasks[i]->counter == 0) {
      ready_queue.push(task_manager->tasks[i]);
    }
  }

  // Step 5: perform simulation

  float sim_time = 0.0f;
  std::map<Device *, float> device_times;
  // map<Device*, SimTask*> device_schedule;
  size_t idx = 0;
  while (!ready_queue.empty()) {
    // Find the task with the earliest start time
    SimTask *cur_task = ready_queue.top();
    ready_queue.pop();
    float ready_time = 0;
    float end_time;
    if (device_times.find(cur_task->device) != device_times.end()) {
      ready_time = device_times[cur_task->device];
    }
    float start_time = std::max(ready_time, cur_task->ready_time);
    if (cur_task->type == SimTask::TASK_NOMINAL_COMM) {
      if (!segment_transfer) {
        end_time = route_transfer(cur_task, start_time, device_times);
      } else {
        bool finished;
        end_time =
            route_transfer_seg(cur_task, start_time, device_times, finished);
        if (!finished) {
          ready_queue.push(cur_task);
          continue;
        }
      }
    } else if (cur_task->type == SimTask::TASK_ALLREDUCE) {
      expand_allreduce(cur_task, start_time, ready_queue);
      idx++;
      continue;
    } else {
      end_time = start_time + cur_task->run_time;
      device_times[cur_task->device] = end_time;
    }

#ifdef DEBUG_PRINT
    printf("task[%lu/%lu] type(%d) run_time(%.4lf) ready_time(%.4lf) "
           "start_time(%.4lf) device(%s)\n",
           idx,
           task_manager->global_task_id,
           cur_task->type,
           cur_task->run_time,
           ready_time,
           start_time,
           (cur_task->device->name).c_str());
#endif

    if (end_time > sim_time) {
      sim_time = end_time;
    }

    for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
      SimTask *next = cur_task->next_tasks[i];
      // next->ready_time = max(next->ready_time, end_time);
      if (end_time > next->ready_time) {
        next->ready_time = end_time;
        // next->prev = t;
      }
      next->counter--;
      if (next->counter == 0) {
        ready_queue.push(next);
      }
    }
    idx++;
  }
  assert(idx == task_manager->global_task_id);

  // Step 6: add penalty to strategies that exceed the memory limits on devices
  // std::vector<size_t> gpu_mem_usage(machine->get_num_gpus(), 0);
  // float memory_penalty = 0.0f;
  // for (size_t l = 0; l < model->layers.size(); l++) {
  //   Op* op = model->layers[l];
  //   ParallelConfig config = global.find(op)->second;
  //   CostMetrics cost_metrics = measure_operator_cost(op, config);
  //   size_t memory_requirement = cost_metrics.memory_requirement;
  //   for (int j = 0; j < config.num_parts(); j++) {
  //     gpu_mem_usage[config.device_ids[j]] += memory_requirement;
  //   }
  // }
  // if (export_file_name != "") {
  //   for (int i = 0; i < machine->get_num_gpus(); i++) {
  //       printf("Before penalty, dev id %d, usage %zu \n", i,
  //       gpu_mem_usage[i]);
  //   }
  // }
  // // Penalize the total runtiem by 1ms if we exceed the memory budget by 1MB
  // for (int i = 0; i < machine->get_num_gpus(); i++) {
  //   MemDevice* gpu_fb_mem = machine->get_gpu_fb_mem(i);
  //   if (gpu_mem_usage[i] > gpu_fb_mem->capacity and gpu_fb_mem->capacity >=
  //   0)
  //     memory_penalty += (gpu_mem_usage[i] - gpu_fb_mem->capacity) * 1e-6;
  // }
  // if (memory_penalty > 0.0f)
  //  printf("Memory penalty = %.4lf ms\n", memory_penalty);
#ifdef WRITE_NETWORK_TRANSFER
  network_transfer_log.close();
#endif

  return sim_time; //  + memory_penalty;
}

float LogicalTaskgraphBasedSimulator::simulate_runtime(
    FFModel const *model,
    std::map<Op const *, ParallelConfig> const &global,
    CompMode comp_mode) {
  return this->simulate_runtime(model, global, comp_mode, "");
}

float LogicalTaskgraphBasedSimulator::route_transfer(
    SimTask *transfer_task,
    float start_time,
    std::map<Device *, float> &device_times) {
  std::vector<CommDevice *> route =
      static_cast<NominalCommDevice *>(transfer_task->device)
          ->expand_to_physical();

  float curr_task_start_time;
  float curr_task_finish_time;
  float curr_task_run_time = 0;
  float curr_task_ready_time = transfer_task->ready_time;
  float xfer_size = transfer_task->xfer_size;

  float final_start_time = 0;
  float final_finish_time = 0;

  SimTask *info_holder = new SimTask();
  info_holder->type = SimTask::TASK_COMM;

  for (unsigned int i = 0; i < route.size(); i++) {
    CommDevice *latency_task_device = route[i];
    if (device_times.find(latency_task_device) == device_times.end()) {
      device_times[latency_task_device] = 0;
    }
    float latency_task_run_time = machine->get_inter_node_gpu_latency();
    float latency_task_ready_time;
    float latency_task_start_time;
    if (i == 0) {
      latency_task_ready_time = curr_task_ready_time + curr_task_run_time;
      latency_task_start_time =
          std::max(device_times[latency_task_device], latency_task_ready_time);
      final_start_time = latency_task_start_time;
    } else {
      latency_task_ready_time = curr_task_finish_time;
      latency_task_start_time =
          std::max(device_times[latency_task_device], latency_task_ready_time);
    }
    float latency_task_finish_time =
        latency_task_start_time + latency_task_run_time;
    device_times[latency_task_device] = latency_task_finish_time;
    float dram_to_dram_run_time = xfer_size / latency_task_device->bandwidth;

    float dram_to_dram_start_time = latency_task_finish_time;
    float dram_to_dram_finish_time =
        dram_to_dram_start_time + dram_to_dram_run_time;
    device_times[latency_task_device] = dram_to_dram_finish_time;

    if (dram_to_dram_finish_time > final_finish_time) {
      final_finish_time = dram_to_dram_finish_time;
    }

    curr_task_ready_time = latency_task_ready_time;
    curr_task_start_time = latency_task_start_time;
    curr_task_finish_time = latency_task_finish_time;
    curr_task_run_time = latency_task_run_time;

#ifdef DEBUG_PRINT
    printf("\texpand: route[%u] run_time(%.4lf) ready_time(%.4lf) "
           "start_time(%.4lf) device(%s)\n",
           i,
           curr_task_run_time,
           curr_task_ready_time,
           curr_task_start_time,
           (latency_task_device->name).c_str());
    printf("\t\td2d: run_time(%.4lf) start_time(%.4lf) device(%s)\n",
           dram_to_dram_run_time,
           dram_to_dram_start_time,
           (latency_task_device->name).c_str());
#endif

    info_holder->device = latency_task_device;
    info_holder->run_time = dram_to_dram_run_time;
    info_holder->xfer_size = xfer_size;
  }
  delete info_holder;

#ifdef WRITE_NETWORK_TRANSFER
  auto *nw = static_cast<NominalCommDevice *>(transfer_task->device);
  network_transfer_log << nw->device_id / machine->get_total_devs() << ", "
                       << nw->device_id % machine->get_total_devs() << ", "
                       << xfer_size << ", " << final_start_time << ", "
                       << final_finish_time << std::endl;
#endif

  transfer_task->run_time = final_finish_time - final_start_time;
  return final_finish_time;
}

float LogicalTaskgraphBasedSimulator::route_transfer_seg(
    SimTask *transfer_task,
    float start_time,
    std::map<Device *, float> &device_times,
    bool &finished) {
  std::vector<CommDevice *> route =
      static_cast<NominalCommDevice *>(transfer_task->device)
          ->expand_to_physical();

  float curr_task_start_time;
  float curr_task_finish_time;
  float curr_task_run_time = 0;
  float curr_task_ready_time = transfer_task->ready_time;
  float xfer_size = transfer_task->xfer_left > segment_size
                        ? segment_size
                        : transfer_task->xfer_left;
  transfer_task->xfer_left = transfer_task->xfer_left > segment_size
                                 ? transfer_task->xfer_left - segment_size
                                 : 0;
  finished = transfer_task->xfer_left == 0;
  // #ifdef DEBUG_PRINT
  // std::cerr << "xfer_total: " << transfer_task->xfer_size << ", xfer_left: "
  // << transfer_task->xfer_left << " finished:" << finished << std::endl;
  // #endif

  float final_start_time = 0;
  float final_finish_time = 0;
  float final_first_seg_finish_time = 0;

  SimTask *info_holder = new SimTask();
  info_holder->type = SimTask::TASK_COMM;

  for (unsigned int i = 0; i < route.size(); i++) {
    CommDevice *latency_task_device = route[i];
    if (device_times.find(latency_task_device) == device_times.end()) {
      device_times[latency_task_device] = 0;
    }
    float latency_task_run_time = machine->get_inter_node_gpu_latency();
    float latency_task_ready_time;
    float latency_task_start_time;
    if (i == 0) {
      latency_task_ready_time = curr_task_ready_time + curr_task_run_time;
      latency_task_start_time =
          std::max(device_times[latency_task_device], latency_task_ready_time);
      final_start_time = latency_task_start_time;
    } else {
      latency_task_ready_time = curr_task_finish_time;
      latency_task_start_time =
          std::max(device_times[latency_task_device], latency_task_ready_time);
    }
    float latency_task_finish_time =
        latency_task_start_time + latency_task_run_time;
    device_times[latency_task_device] = latency_task_finish_time;
    float dram_to_dram_run_time = xfer_size / latency_task_device->bandwidth;
    // std::cerr << "latency_task_device->bandwidth: " <<
    // latency_task_device->bandwidth << std::endl; std::cerr << "d2drt: " <<
    // dram_to_dram_run_time << std::endl;

    float dram_to_dram_start_time = latency_task_finish_time;
    float dram_to_dram_finish_time =
        dram_to_dram_start_time + dram_to_dram_run_time;
    if (i == 0) {
      final_first_seg_finish_time = dram_to_dram_finish_time;
    }
    device_times[latency_task_device] = dram_to_dram_finish_time;

    if (dram_to_dram_finish_time > final_finish_time) {
      final_finish_time = dram_to_dram_finish_time;
    }

    curr_task_ready_time = latency_task_ready_time;
    curr_task_start_time = latency_task_start_time;
    curr_task_finish_time = latency_task_finish_time;
    curr_task_run_time = latency_task_run_time;

#ifdef DEBUG_PRINT
    printf("\texpand: route[%u] run_time(%.4lf) ready_time(%.4lf) "
           "start_time(%.4lf) device(%s)\n",
           i,
           curr_task_run_time,
           curr_task_ready_time,
           curr_task_start_time,
           (latency_task_device->name).c_str());
    printf("\t\td2d: run_time(%.4lf) start_time(%.4lf) device(%s)\n",
           dram_to_dram_run_time,
           dram_to_dram_start_time,
           (latency_task_device->name).c_str());
#endif
    info_holder->device = latency_task_device;
    info_holder->run_time = dram_to_dram_run_time;
    info_holder->xfer_size = xfer_size;
  }
  delete info_holder;

#ifdef WRITE_NETWORK_TRANSFER
  auto *nw = static_cast<NominalCommDevice *>(transfer_task->device);
  network_transfer_log << nw->device_id / machine->get_total_devs() << ", "
                       << nw->device_id % machine->get_total_devs() << ", "
                       << xfer_size << ", " << final_start_time << ", "
                       << final_finish_time << std::endl;
#endif
  if (!finished) {
#ifdef DEBUG_PRINT
    std::cerr << "ready time: " << transfer_task->ready_time << " to "
              << final_first_seg_finish_time << std::endl;
#endif
    transfer_task->ready_time = final_first_seg_finish_time;
  }

  transfer_task->run_time = final_finish_time - final_start_time;
  return final_finish_time;
}

void LogicalTaskgraphBasedSimulator::expand_allreduce(
    SimTask *allreduce_task,
    float start_time,
    std::priority_queue<SimTask *, std::vector<SimTask *>, SimTaskCompare>
        &ready_queue) {

  int n_participants = allreduce_task->next_tasks.size();
  if (n_participants == 1) {
    return;
  }

  SimTask *final_task = new_update_task_unrecorded();

#ifdef FF_USE_NCCL
  // recall that next_task stores node group in this case
  final_task->device = machine->get_gpu(
      reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  MemDevice *src_mem = machine->get_gpu_fb_mem(
      reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  MemDevice *dst_mem;
  // std::cerr << "expand_ar size: " << allreduce_task->xfer_size << ", " <<
  // "grp size: " << n_participants << std::endl;

  int dir = std_uniform(gen) < 0.5 ? 1 : -1;
  // std::cerr << "dir: " << dir << std::endl;
  int round = 0, i = 0;
  // for (int i = 0; i < n_participants; i++) {
  while (round != n_participants) {
    dst_mem = machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(
        allreduce_task->next_tasks[MOD(i + dir, n_participants)]));
    // dst_mem =
    // machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[(i+1)%n_participants]));
    std::vector<CommDevice *> path = machine->get_comm_path(src_mem, dst_mem);
    // if (dir)
    // std::vector<CommDevice *> path = machine->get_comm_path(src_mem,
    // dst_mem); std::cerr << "\tDevices: ";
    for (CommDevice *d : path) {
      SimTask *task = new_comm_task_unrecorded();
      task->device = d;
      // std::cerr << "dir: " << dir << ", " << d->name << ", ";
      task->run_time = 0;
      task->ready_time = allreduce_task->ready_time;
      task->xfer_size = (2.0 * (n_participants - 1)) *
                        allreduce_task->xfer_size / n_participants;
      task->xfer_left = task->xfer_size;
      task->add_next_task(final_task);
      ready_queue.push(task);
    }
    // std::cerr << std::endl;
    src_mem = dst_mem;
    round++;
    i += dir;
  }
  if (final_task->counter == 0) {
    final_task->ready_time = allreduce_task->ready_time;
    ready_queue.push(final_task);
  }
#else
  // assume parameter server in this case
  MemDevice *leader_mem = machine->get_gpu_fb_mem(
      reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  MemDevice *worker_mem;
  SimTask *ps_update_task = new_update_task_unrecorded();
  ps_update_task->device = machine->get_gpu(
      reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  final_task->device = machine->get_gpu(
      reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  ps_update_task->add_next_task(final_task);

  // ps gather
  for (int i = 0; i < n_participants; i++) {
    worker_mem = machine->get_gpu_fb_mem(
        reinterpret_cast<uint64_t>(allreduce_task->next_tasks[i]));
    std::vector<CommDevice *> path =
        machine->get_comm_path(worker_mem, leader_mem);
    for (CommDevice *d : path) {
      SimTask *task = new_comm_task_unrecorded();
      task->device = d;
      task->run_time = 0;
      task->ready_time = allreduce_task->ready_time;
      task->xfer_size = allreduce_task->xfer_size;
      task->xfer_left = task->xfer_size;
      task->add_next_task(ps_update_task);
      ready_queue.push(task);
    }
  }

  // scatter
  for (int i = 0; i < n_participants; i++) {
    worker_mem = machine->get_gpu_fb_mem(
        reinterpret_cast<uint64_t>(allreduce_task->next_tasks[i]));
    std::vector<CommDevice *> path =
        machine->get_comm_path(leader_mem, worker_mem);
    for (CommDevice *d : path) {
      SimTask *task = new_comm_task_unrecorded();
      task->device = d;
      task->run_time = 0;
      task->ready_time = allreduce_task->ready_time;
      task->xfer_size = allreduce_task->xfer_size;
      ps_update_task->add_next_task(task);
      task->add_next_task(final_task);
    }
  }

  if (ps_update_task->counter == 0) {
    assert(final_task->counter == 1);
    ps_update_task->ready_time = allreduce_task->ready_time;
    ready_queue.push(ps_update_task);
  }

#endif
}

SimTask *LogicalTaskgraphBasedSimulator::new_comm_task_unrecorded() {
  SimTask *task = task_manager->new_task();
  task->type = SimTask::TASK_NOMINAL_COMM;
  task->store = false;
  return task;
}

SimTask *LogicalTaskgraphBasedSimulator::new_update_task_unrecorded() {
  SimTask *task = task_manager->new_task();
  task->type = SimTask::TASK_UPDATE;
  task->store = false;
  return task;
}

void LogicalTaskgraphBasedSimulator::add_task_dependencies_with_xfer(
    SimTask *src_task, SimTask *dst_task, size_t message_size) {
  std::vector<CommDevice *> path =
      machine->get_comm_path(src_task->mem, dst_task->mem);
#ifdef DEBUG_PRINT
  // print the communication path
  // printf("Path from %s to %s is: ", src_task->mem->name.c_str(),
  // dst_task->mem->name.c_str()); for (size_t i = 0; i < path.size(); i++) {
  //   printf("%s ", path[i]->name.c_str());
  // }
  // printf("\n");
#endif

  if (path.empty()) {
    src_task->add_next_task(dst_task);
    return;
  }
  assert(message_size > 0);
  std::vector<SimTask *> final_tasks;
  for (CommDevice *d : path) {
    SimTask *task = task_manager->new_nominal_comm_task();
    task->device = d;
    task->run_time = 0;
    task->xfer_size = message_size;
    task->xfer_left = message_size;
    if (!final_tasks.empty()) {
      final_tasks.back()->add_next_task(task);
    }
    final_tasks.push_back(task);
  }
  src_task->add_next_task(final_tasks[0]);
  final_tasks.back()->add_next_task(dst_task);
}

SimTask *TaskManager::new_allreduce_task(Op const *op,
                                         std::vector<int> const &node_ids,
                                         size_t message_size) {
  SimTask *task = new_task();
  task->type = SimTask::TASK_ALLREDUCE;
  // task->counter = node_ids[0];
  for (size_t i = 0; i < node_ids.size(); i++) {
    task->next_tasks.push_back(reinterpret_cast<SimTask *>(node_ids[i]));
  }
  task->xfer_size = message_size;
  return task;
}

SimTask *TaskManager::new_nominal_comm_task() {
  SimTask *task = new_task();
  task->type = SimTask::TASK_NOMINAL_COMM;
  return task;
}

SimTask *TaskManager::new_nominal_comm_task(std::string const &name,
                                            CommDevice *comm_device,
                                            size_t message_size) {
  SimTask *task = new_task();
  task->type = SimTask::TASK_NOMINAL_COMM;
  task->name = name;
  task->device = comm_device;
  task->run_time = comm_device->latency + message_size / comm_device->bandwidth;
  return task;
}
}; // namespace FlexFlow

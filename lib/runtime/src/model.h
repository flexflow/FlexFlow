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
#ifndef _FLEXFLOW_MODEL_H_
#define _FLEXFLOW_MODEL_H_
#include "runtime/config.h"
#include "accessor.h"
#include "utils/graph/node.h"
#include "op-attrs/operator_attrs.h"
#include "utils/hash-utils.h"
#include "utils/tuple.h"
#include "initializer.h"
#include "layer.h"
#include "legion.h"
#include "loss_functions.h"
#include "metrics_functions/metrics_functions.h"
#include "optimizer.h"
#include "parallel_tensor.h"
#include "recompile.h"
#include "simulator.h"
#include "tensor.h"
#include "tl/optional.hpp"
#include <functional>
#include <unistd.h>
#include <utility>
#include "compiler/compiler.h"
#include "op-attrs/ffconst.h"
#include "layer_id.h"
#include "kernels/ff_handler.h"
#include "op_node.h"
#include "op-attrs/tensor_shape.h"
#include "legion_parallel_tensor_shape.h"
#include "index_space_manager.h"
#include "parallel_tensor_uses.h"
#include "computation_graph.h"
#include "parallel_computation_graph.h"
#include "tensor_mapping.h"
#include "operator.h"
#include "parallel_tensor_legion_backing.h"

namespace FlexFlow {

enum ShardingID {
  DataParallelShardingID = 135,
};

class ElementBinary;
class ElementUnary;

MachineView get_basic_data_parallel_machine_view(int num_parts, int dims);

class FFModel {
public:
  FFModel(FFConfig const &config, ComputationGraph const &, ParallelComputationGraph const &);

  optional<ParallelTensor> get_parallel_tensor_from_tensor(Tensor const &tensor) const;

  // ========================================
  // Graph APIs
  // ========================================
  static void register_all_machine_views(int num_nodes,
                                         int gpus_per_node,
                                         int cpus_per_node,
                                         std::vector<MachineView> &valid_views);

  Node get_or_create_fused_parallel_node(
      ParallelTensor const &input,
      std::vector<ParallelOpInfo> const &parallel_ops);
  Node get_or_create_parallel_op_node(ParallelTensor const &input,
                                           ParallelOpInfo const &);
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================

  static PerfMetrics
      update_metrics_task(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  void reset_metrics();
  void init_operators();
  void prefetch();
  void forward(int seq_length = -1);
  void compute_metrics();
  void get_metrics();
  void backward(int seq_length = -1);
  void update();
  bool apply_fusion(std::vector<Op *> const &operators,
                    std::vector<Op *> &new_operators);
  Operator get_final_operator() const;
  void compile(LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void compile(Optimizer *optimizer,
               LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  /* SearchSolution graph_optimize(ComputationGraph const &, */ 
  /*                               MachineSpecification const &); */
#ifdef FF_USE_NCCL
  ncclComm_t *find_nccl_comms(MachineView const &view) const;
#endif
  void recompile_on_condition(RecompileState &r);
  void zero_gradients();

  // APIs for setting iteration configs
public:
  void set_iteration_config_sequence_length(int seq_length);
private:
  void execute_graph_optimize();
  void perform_inplace_optimizations();
  void perform_fusion_optimizations();
  void initialize_nccl_communicators();
  void optimize_unnecessary_gradient_calculations();
  void print_operator_regions() const;
  void create_label_tensor(LossType);
  void populate_tensor_to_parallel_tensor_mapping();

  std::vector<Op *> get_operators();
  std::vector<Op const *> get_operators() const;
public:
  size_t op_global_guid = OP_GUID_FIRST_VALID;
  FFConfig config;
  FFIterationConfig iter_config;
  Optimizer *optimizer = nullptr;
  optional<Loss> loss_op = nullopt;
  optional<Metrics> metrics_op = nullopt;
  std::unique_ptr<Simulator> simulator = nullptr;
  int metrics_input;
  optional<ParallelTensor> parallel_label_tensor;
  optional<Tensor> label_tensor;

  IndexSpaceManager index_space_mgr;
  ParallelTensorManager parallel_tensor_mgr;
  OpNodeManager op_node_mgr;
  ComputationGraph computation_graph;
  ParallelComputationGraph pcg;
  ParallelTensorUses uses;
  TensorMapping tensor_map;

  std::unordered_map<parallel_tensor_guid_t, ParallelTensorLegionBacking> legion_backing;

  std::vector<ParallelTensor> parameters;
  std::vector<FFHandler> handlers;
  Legion::Future current_metrics;
  // Cached operators: key: operator hash, value: operator pointer
  /* std::unordered_map<PCGOperatorAttrs, Op*> cached_ops; */
  std::vector<MachineView> all_valid_views;
#ifdef FF_USE_NCCL
  std::unordered_map<size_t, ncclComm_t *> view_hash_to_nccl_comms;
#endif
private:
  bool debug;
  std::vector<std::unique_ptr<Op>> operators;

  Tensor binary(OperatorType op,
                Tensor const x,
                Tensor const y,
                bool inplace_a = false,
                char const *name = NULL);
  ElementBinary *binary(OperatorType op, char const *name = NULL);
  Tensor unary(OperatorType op,
               Tensor const x,
               bool inplace = true,
               char const *name = NULL,
               float scalar = 0.0);
  ElementUnary *
      unary(OperatorType op, char const *name = NULL, float scalar = 0.0);
  OpNode new_node(Op const *);
};


void top_level_task(Legion::Task const *task,
                    std::vector<Legion::PhysicalRegion> const &regions,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);

void data_load_task(Legion::Task const *task,
                    std::vector<Legion::PhysicalRegion> const &regions,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);

void register_custom_tasks();

}

#endif

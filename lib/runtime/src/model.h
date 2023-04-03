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
#include "tensor_shape.h"
#include "legion_parallel_tensor_shape.h"
#include "index_space_manager.h"
#include "parallel_tensor_manager.h"

namespace FlexFlow {

enum ShardingID {
  DataParallelShardingID = 135,
};

class ElementBinary;
class ElementUnary;

MachineView get_basic_data_parallel_machine_view(int num_parts, int dims);

class FFModel {
public:
  FFModel(FFConfig const &config);

  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(Tensor const &x, char const *name = NULL);
  // Add an add layer
  Tensor add(Tensor const &x,
             Tensor const &y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a subtract layer
  Tensor subtract(Tensor const &x,
                  Tensor const &y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a multiply layer
  Tensor multiply(Tensor const &x,
                  Tensor const &y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a divide layer
  Tensor divide(Tensor const &x,
                Tensor const &y,
                bool inplace_a = false,
                char const *name = NULL);
  // Add a max layer
  Tensor max(Tensor const &x,
             Tensor const &y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a min layer
  Tensor min(Tensor const &x,
             Tensor const &y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a rsqrt layer
  Tensor rsqrt(Tensor const &x, bool inplace = true, char const *name = NULL);
  // Add a pow layer
  Tensor pow(Tensor const &x,
             float const exponent,
             bool inplace = true,
             char const *name = NULL);
  // Add a scalar multiply layer
  Tensor scalar_multiply(Tensor const &x,
                         float const scalar,
                         bool inplace = true,
                         char const *name = NULL);
  Tensor scalar_add(Tensor const &x,
                    float const scalar,
                    bool inplace = true,
                    char const *name = NULL);
  Tensor scalar_sub(Tensor const &x,
                    float const scalar,
                    bool inplace = true,
                    char const *name = NULL);
  Tensor scalar_truediv(Tensor const &x,
                        float const scalar,
                        bool inplace = true,
                        char const *name = NULL);
  // Add a sin layer
  Tensor sin(Tensor const &x, char const *name = NULL);
  // Add a cos layer
  Tensor cos(Tensor const &x, char const *name = NULL);
  // Add an activation layer
  Tensor relu(Tensor const &x, bool inplace = true, char const *name = NULL);
  Tensor identity(Tensor const &x, char const *name = NULL);
  Tensor gelu(Tensor const &x, char const *name = NULL);
  Tensor sigmoid(Tensor const &x, char const *name = NULL);
  Tensor tanh(Tensor const &x, char const *name = NULL);
  Tensor elu(Tensor const &x, bool inplace = true, char const *name = NULL);
  // Add a 2D convolutional layer
  Tensor conv2d(Tensor const &input,
                int outChannels,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                ActiMode activation = AC_MODE_NONE,
                int groups = 1,
                bool use_bias = true,
                Layer const *shared_op = NULL,
                Initializer *krenel_initializer = NULL,
                Initializer *bias_initializer = NULL,
                char const *name = NULL);
  // Add a dropout layer
  Tensor dropout(Tensor const &input,
                 float rate,
                 unsigned long long seed = 0,
                 char const *name = NULL);
  // Add an embedding layer
  Tensor embedding(Tensor const &input,
                   int num_entires,
                   int outDim,
                   AggrMode aggr,
                   DataType dtype = DT_FLOAT,
                   Layer const *shared_op = NULL,
                   Initializer *kernel_initializer = NULL,
                   char const *name = NULL);
  // Add a gather layer
  Tensor gather(Tensor const &input,
                Tensor const &index,
                int dim,
                char const *name = NULL);
  // Add a group_by layer
  void group_by(Tensor const &data,
                Tensor const &assign,
                Tensor *outputs,
                int n,
                float alpha,
                char const *name = NULL);
  // Add a cache layer
  Tensor cache(Tensor const &input,
               int num_batches,
               std::function<float(float *, void const *, void const *, int)>
                   score_f = {},
               char const *name = NULL);
  // Add aggregate layer
  Tensor aggregate(Tensor const *inputs,
                   int n,
                   float lambda_bal,
                   char const *name = NULL);
  // Add aggregate_spec layer
  Tensor aggregate_spec(Tensor const *inputs,
                        int n,
                        float lambda_bal,
                        char const *name = NULL);
  // Add a 2D pooling layer
  Tensor pool2d(Tensor const &input,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE,
                char const *name = NULL);
  // Add a batch_norm layer
  Tensor layer_norm(Tensor const &input,
                    std::vector<int> const &axes,
                    bool elementwise_affine,
                    float eps,
                    char const *name = NULL);
  // Add a batch_norm layer
  Tensor
      batch_norm(Tensor const &input, bool relu = true, char const *name = NULL);
  // Add a batch_matmul layer
  Tensor batch_matmul(Tensor const &A,
                      Tensor const &B,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      char const *name = nullptr);
  // Add a dense layer
  Tensor dense(Tensor const &input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               DataType data_type = DT_FLOAT,
               Layer const *shared_op = NULL,
               Initializer *kernel_initializer = NULL,
               Initializer *bias_initializer = NULL,
               char const *name = NULL);
  // Add a cast layer
  Tensor cast(Tensor const &input, DataType dtype, char const *name = nullptr);
  // Add a concat layer
  Tensor
      concat(int n, Tensor const *tensors, int axis, char const *name = NULL);
  // Add a mean layer
  Tensor mean(Tensor const &input,
              std::vector<int> const &dims,
              bool keepdims,
              char const *name);
  // Add a moe layer (wrapping topk, group_by and aggregate operators)
  Tensor moe(Tensor const &input,
             int num_exp,
             int num_select,
             int expert_hidden_size,
             float alpha,
             float lambda);
  // Add a split layer
  void split(Tensor const &input,
             Tensor *outputs,
             std::vector<int> const &split,
             int axis,
             char const *name = NULL);
  // Add a flat layer
  Tensor flat(Tensor const &input, char const *name = NULL);
  // Add a softmax layer
  Tensor softmax(Tensor const &input, int dim = -1, char const *name = NULL);
  // Create input tensors and constants
  Tensor transpose(Tensor const &input,
                   std::vector<int> const &perm,
                   char const *name = NULL);
  Tensor reduce_sum(Tensor const &input,
                    std::vector<int> const &axes,
                    bool keepdims = false,
                    char const *name = nullptr);
  Tensor reshape(Tensor const &input,
                 std::vector<int> const &shape,
                 char const *name = NULL);
  Tensor reverse(Tensor const &input, int axis, char const *name = NULL);
  void top_k(Tensor const &input,
             Tensor *outputs,
             int k,
             bool sorted,
             char const *name = NULL);
  Tensor multihead_attention(Tensor const &query,
                             Tensor const &key,
                             Tensor const &value,
                             int embed_dim,
                             int num_heads,
                             int kdim = 0,
                             int vdim = 0,
                             float dropout = 0.0f,
                             bool bias = true,
                             bool add_bias_kv = false,
                             bool add_zero_attn = false,
                             Initializer *kernel_initializer = NULL,
                             char const *name = NULL);
  Tensor create_tensor(LegionTensorShape const &shape,
                       Layer const *owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  ParallelTensor
      create_parallel_tensor(LegionParallelTensorShape const &,
                             Op const *owner_op = NULL,
                             int owner_idx = 0,
                             bool create_grad = true,
                             size_t input_tensor_guid = 0);
  Tensor create_tensor(TensorShape const &,
                       Layer const *owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  ParallelTensor create_parallel_tensor(ParallelTensorShape const &,
                                        Op const *owner_op = NULL,
                                        int owner_idx = 0,
                                        bool create_grad = true,
                                        size_t input_tensor_guid = 0);
  Parameter create_weight(
      TensorShape const &,
      Layer const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  Parameter create_weight(
      LegionTensorShape const &,
      Layer const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  ParallelParameter create_parallel_weight(
      ParallelTensorShape const &,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  ParallelParameter create_parallel_weight(
      LegionParallelTensorShape const &,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);

  optional<ParallelTensor> get_parallel_tensor_from_tensor(Tensor const &tensor) const;

  template <int NDIM>
  Tensor create_constant(int const dims[], float value, DataType date_type);
  // ========================================
  // Graph APIs
  // ========================================
  bool convert_graph_to_operators(SearchSolution const &);
  static void register_all_machine_views(int num_nodes,
                                         int gpus_per_node,
                                         int cpus_per_node,
                                         std::vector<MachineView> &valid_views);

  Node get_or_create_noop_node(ParallelTensor const &input);
  Node get_or_create_input_node(ParallelTensorShape const &);
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
  Op *get_final_operator() const;
  void compile(LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void compile(Optimizer *optimizer,
               LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  SearchSolution graph_optimize(ComputationGraph const &, 
                                MachineSpecification const &);
#ifdef FF_USE_NCCL
  ncclComm_t *find_nccl_comms(MachineView const &view) const;
#endif
  void recompile_on_condition(RecompileState &r);
  void zero_gradients();

  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>>
      get_bwd_edge_map() const;

  void create_operators_from_layers();
  Op *create_operator_from_layer(Layer *layer,
                                 std::vector<ParallelTensor> const &inputs);
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

  template <int NDIM>
  ParallelParameter create_parallel_weight(
      ParallelTensorShape const &,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);

  template <int NDIM>
  Tensor create_tensor(TensorShape const &,
                       Layer const *owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  template <int NDIM>
  ParallelTensor create_parallel_tensor(ParallelTensorShape const &,
                                        Op const *owner_op = NULL,
                                        int owner_idx = 0,
                                        bool create_grad = true,
                                        size_t input_tensor_guid = 0);

public:
  size_t op_global_guid;
  size_t node_global_guid;
  FFConfig config;
  FFIterationConfig iter_config;
  Optimizer *optimizer;
  optional<Loss> loss_op;
  optional<Metrics> metrics_op;
  std::unique_ptr<Simulator> simulator;
  int metrics_input;
  optional<ParallelTensor> parallel_label_tensor;
  optional<Tensor> label_tensor;
  IndexSpaceManager index_space_mgr;
  ParallelTensorManager parallel_tensor_mgr;
  TensorManager tensor_mgr;
  LayerManager layer_mgr;

  std::vector<Op *> operators;
  std::vector<ParallelTensor> parameters;
  std::vector<FFHandler> handlers;
  Legion::Future current_metrics;
  // Cached operators: key: operator hash, value: operator pointer
  std::unordered_map<PCGOperatorAttrs, Op*> cached_ops;
  std::vector<MachineView> all_valid_views;
#ifdef FF_USE_NCCL
  std::unordered_map<size_t, ncclComm_t *> view_hash_to_nccl_comms;
#endif
private:
  bool debug;

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
  OpNode new_node(Op *);
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

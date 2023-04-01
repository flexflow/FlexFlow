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

namespace FlexFlow {

enum ShardingID {
  DataParallelShardingID = 135,
};

// namespace Legion {
//   class Serializer;
// }


class FFModel;
class ParallelOp;
class ElementBinary;
class ElementUnary;

class NoOp;

MachineView get_basic_data_parallel_machine_view(int num_parts, int dims);

class FFModel {
public:
  FFModel(FFConfig &config);

  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(const Tensor x, char const *name = NULL);
  // Add an add layer
  Tensor add(const Tensor x,
             const Tensor y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a subtract layer
  Tensor subtract(const Tensor x,
                  const Tensor y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a multiply layer
  Tensor multiply(const Tensor x,
                  const Tensor y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a divide layer
  Tensor divide(const Tensor x,
                const Tensor y,
                bool inplace_a = false,
                char const *name = NULL);
  // Add a max layer
  Tensor max(const Tensor x,
             const Tensor y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a min layer
  Tensor min(const Tensor x,
             const Tensor y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a rsqrt layer
  Tensor rsqrt(const Tensor x, bool inplace = true, char const *name = NULL);
  // Add a pow layer
  Tensor pow(const Tensor x,
             float const exponent,
             bool inplace = true,
             char const *name = NULL);
  // Add a scalar multiply layer
  Tensor scalar_multiply(const Tensor x,
                         float const scalar,
                         bool inplace = true,
                         char const *name = NULL);
  Tensor scalar_add(const Tensor x,
                    float const scalar,
                    bool inplace = true,
                    char const *name = NULL);
  Tensor scalar_sub(const Tensor x,
                    float const scalar,
                    bool inplace = true,
                    char const *name = NULL);
  Tensor scalar_truediv(const Tensor x,
                        float const scalar,
                        bool inplace = true,
                        char const *name = NULL);
  // Add a sin layer
  Tensor sin(const Tensor x, char const *name = NULL);
  // Add a cos layer
  Tensor cos(const Tensor x, char const *name = NULL);
  // Add an activation layer
  Tensor relu(const Tensor x, bool inplace = true, char const *name = NULL);
  Tensor identity(const Tensor x, char const *name = NULL);
  Tensor gelu(const Tensor x, char const *name = NULL);
  Tensor sigmoid(const Tensor x, char const *name = NULL);
  Tensor tanh(const Tensor x, char const *name = NULL);
  Tensor elu(const Tensor x, bool inplace = true, char const *name = NULL);
  // Add a 2D convolutional layer
  Tensor conv2d(const Tensor input,
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
  Tensor dropout(const Tensor input,
                 float rate,
                 unsigned long long seed = 0,
                 char const *name = NULL);
  // Add an embedding layer
  Tensor embedding(const Tensor input,
                   int num_entires,
                   int outDim,
                   AggrMode aggr,
                   DataType dtype = DT_FLOAT,
                   Layer const *shared_op = NULL,
                   Initializer *kernel_initializer = NULL,
                   char const *name = NULL);
  // Add a gather layer
  Tensor gather(const Tensor input,
                const Tensor index,
                int dim,
                char const *name = NULL);
  // Add a group_by layer
  void group_by(const Tensor data,
                const Tensor assign,
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
  Tensor pool2d(const Tensor input,
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
  Tensor layer_norm(const Tensor input,
                    std::vector<int> const &axes,
                    bool elementwise_affine,
                    float eps,
                    char const *name = NULL);
  // Add a batch_norm layer
  Tensor
      batch_norm(const Tensor input, bool relu = true, char const *name = NULL);
  // Add a batch_matmul layer
  Tensor batch_matmul(const Tensor A,
                      const Tensor B,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      char const *name = nullptr);
  // Add a dense layer
  Tensor dense(const Tensor input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               DataType data_type = DT_FLOAT,
               Layer const *shared_op = NULL,
               Initializer *kernel_initializer = NULL,
               Initializer *bias_initializer = NULL,
               char const *name = NULL);
  // Add a cast layer
  Tensor cast(const Tensor input, DataType dtype, char const *name = nullptr);
  // Add a concat layer
  Tensor
      concat(int n, Tensor const *tensors, int axis, char const *name = NULL);
  // Add a mean layer
  Tensor mean(const Tensor input,
              std::vector<int> const &dims,
              bool keepdims,
              char const *name);
  // Add a moe layer (wrapping topk, group_by and aggregate operators)
  Tensor moe(const Tensor input,
             int num_exp,
             int num_select,
             int expert_hidden_size,
             float alpha,
             float lambda);
  // Add a split layer
  void split(const Tensor input,
             Tensor *outputs,
             std::vector<int> const &split,
             int axis,
             char const *name = NULL);
  // Add a flat layer
  Tensor flat(const Tensor input, char const *name = NULL);
  // Add a softmax layer
  Tensor softmax(const Tensor input, int dim = -1, char const *name = NULL);
  // Create input tensors and constants
  Tensor transpose(const Tensor input,
                   std::vector<int> const &perm,
                   char const *name = NULL);
  Tensor reduce_sum(const Tensor input,
                    std::vector<int> const &axes,
                    bool keepdims = false,
                    char const *name = nullptr);
  Tensor reshape(const Tensor input,
                 std::vector<int> const &shape,
                 char const *name = NULL);
  Tensor reverse(const Tensor input, int axis, char const *name = NULL);
  void top_k(const Tensor input,
             Tensor *outputs,
             int k,
             bool sorted,
             char const *name = NULL);
  Tensor multihead_attention(const Tensor query,
                             const Tensor key,
                             const Tensor value,
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
  Tensor create_tensor_legion_ordering(int num_dim,
                                       int const dims[],
                                       DataType data_type,
                                       Layer const *owner_op = NULL,
                                       int owner_idx = 0,
                                       bool create_grad = true);
  ParallelTensor
      create_parallel_tensor_legion_ordering(int num_dim,
                                             const ParallelDim dims[],
                                             DataType data_type,
                                             Op const *owner_op = NULL,
                                             int owner_idx = 0,
                                             bool create_grad = true,
                                             size_t input_tensor_guid = 0);
  Tensor create_tensor(int num_dim,
                       int const dims[],
                       DataType data_type,
                       Layer const *owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  ParallelTensor create_parallel_tensor(int num_dim,
                                        const ParallelDim dims[],
                                        DataType data_type,
                                        Op const *owner_op = NULL,
                                        int owner_idx = 0,
                                        bool create_grad = true,
                                        size_t input_tensor_guid = 0);
  template <int NDIM>
  Tensor create_tensor(int const dims[],
                       DataType data_type,
                       Layer const *owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  template <int NDIM>
  ParallelTensor create_parallel_tensor(const ParallelDim dims[],
                                        DataType data_type,
                                        Op const *owner_op = NULL,
                                        int owner_idx = 0,
                                        bool create_grad = true,
                                        size_t input_tensor_guid = 0);
  Parameter
      create_weight(int numdim,
                    int const dims[],
                    DataType data_type,
                    Layer const *owner_op = NULL,
                    bool create_grad = true,
                    Initializer *initializer = NULL,
                    ParameterSyncType sync_type = ParameterSyncType::NONE);
  Parameter create_weight_legion_ordering(
      int numdim,
      int const dims[],
      DataType data_type,
      Layer const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  template <int NDIM>
  ParallelParameter create_parallel_weight(
      const ParallelDim dims[],
      DataType data_type,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  ParallelParameter create_parallel_weight(
      int numdim,
      const ParallelDim dims[],
      DataType data_type,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  ParallelParameter create_parallel_weight_legion_ordering(
      int numdim,
      const ParallelDim dims[],
      DataType data_type,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);

  void map_tensor(ParallelTensor tensor, Op const *parallel_op);
  void map_weight(ParallelTensor tensor, Op const *parallel_op);
  bool get_parallel_tensor_from_tensor(const Tensor tensor,
                                       ParallelTensor &parallel_tensor) const;

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
  // ========================================
  // Internal PCG::Node creation APIs
  // ========================================
  /* tl::optional<Node> get_or_create_node(std::vector<ParallelTensor> const &inputs, */
  /*                         PCGOperatorAttrs const &attrs) { */
  /*   auto input_shapes = vector_transform([](ParallelTensor const &t) { return t->get_shape(); }, inputs); */

  /*   if (!is_valid(attrs, input_shapes)) { */
  /*     return tl::nullopt; */
  /*   } */

  /*   T *op = nullptr; */

  /*   std::pair<typename ToShape<typename T::Input>::type, Params> key{ */
  /*       input_shapes, params}; */
  /*   auto &cache = get<std::unordered_map< */
  /*       std::pair<typename ToShape<typename T::Input>::type, Params>, */
  /*       T *>>(this->cached_ops); */
  /*   auto const &it = cache.find(key); */
  /*   if (it != cache.end()) { */
  /*     op = it->second; */
  /*   } else { */
  /*     op = new T(*this, params, input); */
  /*     cache[key] = op; */
  /*   } */

  /*   assert(op->get_params() == params); */
  /*   return this->new_node(op); */
  /* } */

  Node get_or_create_noop_node(const ParallelTensor input);
  Node get_or_create_input_node(ParallelTensorShape const &);
  Node get_or_create_fused_parallel_node(
      const ParallelTensor input,
      std::vector<ParallelOpInfo> const &parallel_ops);
  Node get_or_create_parallel_op_node(const ParallelTensor input,
                                           ParallelOpInfo const &);
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  void create_disjoint_partition(int num_dims,
                                 const ParallelDim dims[],
                                 Legion::IndexSpace const &part_is,
                                 Legion::LogicalRegion const &region,
                                 Legion::LogicalPartition &part);
  template <int NDIM, int TDIM>
  void create_disjoint_partition_with_dim2(
      const ParallelDim dims[],
      Legion::IndexSpaceT<TDIM> const &part_is,
      Legion::LogicalRegion const &region,
      Legion::LogicalPartition &part);
  void create_aliased_partition(int num_dims,
                                const ParallelDim dims[],
                                int aliased_dim,
                                Legion::IndexSpace const &part_is,
                                Legion::LogicalRegion const &region,
                                Legion::LogicalPartition &part);
  template <int NDIM, int TDIM>
  void create_aliased_partition_with_dim2(
      const ParallelDim dims[],
      int aliased_dim,
      Legion::IndexSpaceT<TDIM> const &part_is,
      Legion::LogicalRegion const &region,
      Legion::LogicalPartition &part);

  template <int NDIM>
  void create_disjoint_partition(const ParallelTensor tensor,
                                 Legion::IndexSpaceT<NDIM> const &part_is,
                                 Legion::LogicalPartition &part_fwd,
                                 Legion::LogicalPartition &part_bwd);

  template <int NDIM, int TDIM>
  void create_data_parallel_partition_with_diff_dims(
      const ParallelTensor tensor,
      Legion::IndexSpaceT<TDIM> const &task_is,
      Legion::LogicalPartition &part_fwd,
      Legion::LogicalPartition &part_bwd);
  template <int NDIM>
  void map_conv_weight(ParallelTensor p, Op const *parallel_op);
  template <int NDIM, int TDIM>
  void map_linear_weight(ParallelTensor p, Op const *parallel_op);
  template <int NDIM, int TDIM>
  ParallelTensor create_linear_replica(int const *dims,
                                       Legion::IndexSpaceT<TDIM> const &part_is,
                                       DataType data_type);
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
#ifdef FF_USE_PROPAGATE
  void propagate(std::map<Op *, MachineView> const &current,
                 std::map<Op *, MachineView> &next) const;
#endif
  void rewrite(std::map<Op const *, MachineView> const &current,
               std::map<Op const *, MachineView> &next,
               bool use_propagation) const;
  void recompile_on_condition(RecompileState &r);
  void zero_gradients();
  void print_layers(int id);

  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>>
      get_bwd_edge_map() const;

  // Internal funcitons
  Legion::IndexSpace get_or_create_task_is(MachineView const &);
  Legion::IndexSpace get_or_create_task_is(Legion::Domain const &domain);
  Legion::IndexSpace get_or_create_task_is(const ParallelTensor);
  Legion::IndexSpace get_task_is(Legion::Domain const &domain) const;
  Legion::IndexSpace get_task_is(MachineView const &) const;
  void create_operators_from_layers();
  Op *create_operator_from_layer(Layer *layer,
                                 std::vector<ParallelTensor> const &inputs);
  // APIs for setting iteration configs
public:
  void set_iteration_config_sequence_length(int seq_length);

public:
  size_t op_global_guid, layer_global_guid;
  size_t tensor_global_guid, parallel_tensor_global_guid, node_global_guid;
  FFConfig config;
  FFIterationConfig iter_config;
  Optimizer *optimizer;
  Loss *loss_op;
  Metrics *metrics_op;
  Simulator *simulator;
  int metrics_input;
  ParallelTensor parallel_label_tensor;
  Tensor label_tensor;

  std::vector<Layer *> layers;
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
  std::map<MachineView, Legion::IndexSpace> all_task_is;

  template <int NDIM>
  void map_tensor_with_dim(ParallelTensor tensor, Op const *parallel_op);
  template <int NDIM, int TDIM>
  void map_tensor_with_dim2(ParallelTensor tensor, Op const *parallel_op);
  template <int NDIM>
  void map_weight_with_dim(ParallelTensor weight, Op const *parallel_op);

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
  Node new_node(Op *);
};

class UtilityTasks {
public:
  static FFHandler
      init_cuda_task(Legion::Task const *task,
                     std::vector<Legion::PhysicalRegion> const &regions,
                     Legion::Context ctx,
                     Legion::Runtime *runtime);
  static void dummy_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  static void
      init_images_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      init_labels_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      load_images_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      normalize_images_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
};

void top_level_task(Legion::Task const *task,
                    std::vector<Legion::PhysicalRegion> const &regions,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);

void data_load_task(Legion::Task const *task,
                    std::vector<Legion::PhysicalRegion> const &regions,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);

void register_flexflow_internal_tasks();

void register_custom_tasks();

}; // namespace FlexFlow

#endif //_FLEXFLOW_MODEL_H_

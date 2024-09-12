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
#include "accessor.h"
#include "config.h"
#include "device.h"
#include "flexflow/inference.h"
#include "flexflow/memory_optimization.h"
#include "flexflow/node.h"
#include "flexflow/operator_params.h"
#include "flexflow/utils/hash_utils.h"
#include "flexflow/utils/tuple.h"
#include "initializer.h"
#include "layer.h"
#include "legion.h"
#include "loss_functions.h"
#include "metrics_functions.h"
#include "optimizer.h"
#include "parallel_tensor.h"
#include "recompile.h"
#include "runtime.h"
#include "simulator.h"
#include "tensor.h"
#include "tl/optional.hpp"
#include <functional>
#include <unistd.h>
#include <utility>

#include "ffconst.h"
#include "fftype.h"

namespace FlexFlow {

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FF_INIT_TASK_ID,
  IMAGE_INIT_TASK_ID,
  LABEL_INIT_TASK_ID,
  LOAD_IMAGES_TASK_ID,
  NORMALIZE_IMAGES_TASK_ID,
  ELEMENTBINARY_INIT_TASK_ID,
  ELEMENTBINARY_INF_TASK_ID,
  ELEMENTBINARY_FWD_TASK_ID,
  ELEMENTBINARY_BWD_TASK_ID,
  ELEMENTUNARY_INIT_TASK_ID,
  ELEMENTUNARY_FWD_TASK_ID,
  ELEMENTUNARY_INF_TASK_ID,
  ELEMENTUNARY_BWD_TASK_ID,
  EXPERTS_INIT_TASK_ID,
  EXPERTS_FWD_TASK_ID,
  EXPERTS_BWD_TASK_ID,
  EXPERTS_INF_TASK_ID,
  CONV2D_INIT_TASK_ID,
  CONV2D_INIT_PARA_TASK_ID,
  CONV2D_FWD_TASK_ID,
  CONV2D_BWD_TASK_ID,
  CONV2D_UPD_TASK_ID,
  DROPOUT_INIT_TASK_ID,
  DROPOUT_FWD_TASK_ID,
  DROPOUT_BWD_TASK_ID,
  EMBED_INIT_TASK_ID,
  EMBED_FWD_TASK_ID,
  EMBED_INF_TASK_ID,
  EMBED_BWD_TASK_ID,
  GATHER_INIT_TASK_ID,
  GATHER_FWD_TASK_ID,
  GATHER_BWD_TASK_ID,
  GROUP_BY_INIT_TASK_ID,
  GROUP_BY_FWD_TASK_ID,
  GROUP_BY_BWD_TASK_ID,
  CACHE_INIT_TASK_ID,
  CACHE_FWD_TASK_ID,
  CACHE_UPDATE_TASK_ID,
  CAST_INIT_TASK_ID,
  CAST_FWD_TASK_ID,
  CAST_BWD_TASK_ID,
  AGGREGATE_INIT_TASK_ID,
  AGGREGATE_FWD_TASK_ID,
  AGGREGATE_BWD_TASK_ID,
  AGG_SPEC_INIT_TASK_ID,
  AGG_SPEC_FWD_TASK_ID,
  AGG_SPEC_BWD_TASK_ID,
  POOL2D_INIT_TASK_ID,
  POOL2D_FWD_TASK_ID,
  POOL2D_BWD_TASK_ID,
  BATCHNORM_INIT_TASK_ID,
  BATCHNORM_INIT_PARA_TASK_ID,
  BATCHNORM_FWD_TASK_ID,
  BATCHNORM_BWD_TASK_ID,
  BATCHMATMUL_INIT_TASK_ID,
  BATCHMATMUL_FWD_TASK_ID,
  BATCHMATMUL_BWD_TASK_ID,
  LAYERNORM_INIT_TASK_ID,
  LAYERNORM_FWD_TASK_ID,
  LAYERNORM_INF_TASK_ID,
  LAYERNORM_BWD_TASK_ID,
  RESIDUAL_LAYERNORM_INIT_TASK_ID,
  RESIDUAL_LAYERNORM_INF_TASK_ID,
  ADD_BIAS_RESIDUAL_LAYERNORM_INIT_TASK_ID,
  ADD_BIAS_RESIDUAL_LAYERNORM_INF_TASK_ID,
  SIGMOID_SILU_MULTI_INIT_TASK_ID,
  SIGMOID_SILU_MULTI_INF_TASK_ID,
  LINEAR_INIT_TASK_ID,
  LINEAR_INIT_PARA_TASK_ID,
  LINEAR_INF_TASK_ID,
  LINEAR_FWD_TASK_ID,
  LINEAR_BWD_TASK_ID,
  LINEAR_BWD2_TASK_ID,
  LINEAR_UPD_TASK_ID,
  FLAT_INIT_TASK_ID,
  FLAT_FWD_TASK_ID,
  FLAT_BWD_TASK_ID,
  SOFTMAX_INIT_TASK_ID,
  SOFTMAX_FWD_TASK_ID,
  SOFTMAX_BWD_TASK_ID,
  SOFTMAX_INF_TASK_ID,
  CONCAT_INIT_TASK_ID,
  CONCAT_FWD_TASK_ID,
  CONCAT_BWD_TASK_ID,
  SPLIT_INIT_TASK_ID,
  SPLIT_FWD_TASK_ID,
  SPLIT_BWD_TASK_ID,
  REDUCE_INIT_TASK_ID,
  REDUCE_FWD_TASK_ID,
  REDUCE_BWD_TASK_ID,
  RESHAPE_INIT_TASK_ID,
  RESHAPE_FWD_TASK_ID,
  RESHAPE_BWD_TASK_ID,
  REVERSE_INIT_TASK_ID,
  REVERSE_FWD_TASK_ID,
  REVERSE_BWD_TASK_ID,
  TOPK_INIT_TASK_ID,
  TOPK_FWD_TASK_ID,
  TOPK_BWD_TASK_ID,
  GUMBEL_TOPK_INIT_TASK_ID,
  GUMBEL_TOPK_INF_TASK_ID,
  GUMBEL_TOPK_INF_SPECULATIVE_TASK_ID,
  ARG_TOPK_INIT_TASK_ID,
  ARG_TOPK_INF_TASK_ID,
  ARG_TOPK_INF_SPECULATIVE_TASK_ID,
  SAMPLING_INIT_TASK_ID,
  SAMPLING_INF_TASK_ID,
  ARGMAX_INIT_TASK_ID,
  ARGMAX_BEAM_INF_TASK_ID,
  ARGMAX_NORM_INF_TASK_ID,
  TRANSPOSE_INIT_TASK_ID,
  TRANSPOSE_FWD_TASK_ID,
  TRANSPOSE_BWD_TASK_ID,
  ATTENTION_INIT_TASK_ID,
  ATTENTION_FWD_TASK_ID,
  ATTENTION_BWD_TASK_ID,
  RMSNORM_INIT_TASK_ID,
  RMSNORM_FWD_TASK_ID,
  RMSNORM_INF_TASK_ID,
  RESIDUAL_RMSNORM_INIT_TASK_ID,
  RESIDUAL_RMSNORM_INF_TASK_ID,
  //   BEAM_TOPK_INIT_TASK_ID,
  //   BEAM_TOPK_INF_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_FWD_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_BWD_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
  SPEC_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
  SPEC_INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
  TREE_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
  TREE_INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
  MSELOSS_BWD_TASK_ID,
  FUSEDOP_INIT_TASK_ID,
  FUSEDOP_FWD_TASK_ID,
  FUSEDOP_BWD_TASK_ID,
  FUSEDOP_INF_TASK_ID,
  NOOP_INIT_TASK_ID,
  // Metrics tasks
  METRICS_COMP_TASK_ID,
  UPDATE_METRICS_TASK_ID,
  // Parameter server prefetch task
  PS_PREFETCH_TASK_ID,
  // Loss
  LOSS_BWD_TASK_ID,
  // Optimizer with PS
  SGD_UPD_PS_TASK_ID,
  ADAM_UPD_PS_TASK_ID,
  // Optimizer with NCCL
  SGD_UPD_NCCL_TASK_ID,
  ADAM_UPD_NCCL_TASK_ID,
  // Initializer
  GLOROT_INIT_TASK_ID,
  ZERO_INIT_TASK_ID,
  CONSTANT_INIT_TASK_ID,
  UNIFORM_INIT_TASK_ID,
  NORMAL_INIT_TASK_ID,
  // NCCL tasks
  NCCL_GETUNIQUEID_TASK_ID,
  NCCL_INIT_COMMS_TASK_ID,
  NCCL_FINISH_COMMS_TASK_ID,
  // Search
  STRATEGY_SEARCH_TASK_ID,
  // Graph
  GRAPH_OPTIMIZE_TASK_ID,
  // Python data loader
  PY_DL_FLOAT_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT32_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT64_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_FLOAT_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT32_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT64_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_FLOAT_LOAD_BATCH_GPU_TASK_ID,
  PY_DL_INT32_LOAD_BATCH_GPU_TASK_ID,
  PY_DL_INT64_LOAD_BATCH_GPU_TASK_ID,
  // Parallel Ops
  REPARTITION_INIT_TASK_ID,
  REPARTITION_FWD_TASK_ID,
  REPARTITION_BWD_TASK_ID,
  COMBINE_INIT_TASK_ID,
  COMBINE_FWD_TASK_ID,
  COMBINE_BWD_TASK_ID,
  REPLICATE_INIT_TASK_ID,
  REPLICATE_FWD_TASK_ID,
  REPLICATE_BWD_TASK_ID,
  REDUCTION_INIT_TASK_ID,
  REDUCTION_FWD_TASK_ID,
  REDUCTION_BWD_TASK_ID,
  PIPELINE_INIT_TASK_ID,
  PIPELINE_FWD_TASK_ID,
  PIPELINE_BWD_TASK_ID,
  ALLREDUCE_INIT_TASK_ID,
  ALLREDUCE_INF_TASK_ID,
  ALLREDUCE_FWD_TASK_ID,
  ALLREDUCE_BWD_TASK_ID,
  FUSED_PARALLELOP_INIT_TASK_ID,
  FUSED_PARALLELOP_FWD_TASK_ID,
  FUSED_PARALLELOP_BWD_TASK_ID,
  // InferenceManager & RequestManager
  RM_LOAD_TOKENS_TASK_ID,
  RM_LOAD_POSITION_TASK_ID,
  RM_LOAD_BATCH_CONFIG_TASK_ID,
  RM_GET_NEXT_BATCH_CONFIG_TASK_ID,
  RM_PREPARE_NEXT_BATCH_TASK_ID,
  RM_PREPARE_NEXT_BATCH_INIT_TASK_ID,
  RM_PREPARE_NEXT_BATCH_SPEC_TASK_ID,
  RM_PREPARE_NEXT_BATCH_VERIFY_TASK_ID,
  RM_BACKGROUND_SERVING_TASK_ID,
  // Custom tasks
  CUSTOM_GPU_TASK_ID_FIRST,
  CUSTOM_GPU_TASK_ID_1,
  CUSTOM_GPU_TASK_ID_2,
  CUSTOM_GPU_TASK_ID_3,
  CUSTOM_GPU_TASK_ID_4,
  CUSTOM_GPU_TASK_ID_5,
  CUSTOM_GPU_TASK_ID_6,
  CUSTOM_GPU_TASK_ID_7,
  CUSTOM_GPU_TASK_ID_8,
  CUSTOM_GPU_TASK_ID_LAST,
  CUSTOM_CPU_TASK_ID_FIRST,
  CUSTOM_CPU_TASK_ID_1,
  CUSTOM_CPU_TASK_ID_2,
  CUSTOM_CPU_TASK_ID_3,
  CUSTOM_CPU_TASK_ID_4,
  CUSTOM_CPU_TASK_ID_5,
  CUSTOM_CPU_TASK_ID_6,
  CUSTOM_CPU_TASK_ID_7,
  CUSTOM_CPU_TASK_ID_LAST,
  // Make sure PYTHON_TOP_LEVEL_TASK_ID is
  // consistent with python/main.cc
  PYTHON_TOP_LEVEL_TASK_ID = 11111,
  // Tensor Equal Task
  TENSOR_EQUAL_TASK_ID,
};

enum ShardingID {
  DataParallelShardingID = 135,
};

// namespace Legion {
//   class Serializer;
// }

namespace PCG {
class SearchHelper;
class GraphSearchHelper;
class Graph;
}; // namespace PCG

class FFModel;
class ParallelOp;

void solve_parallel_dim_mappings(
    std::vector<ParallelDimMappingRecord> const &mapping,
    std::vector<ParallelDim const *> const &inputs,
    std::vector<ParallelDim *> const &weights,
    std::vector<ParallelDim *> const &outputs);
std::unordered_map<int, int> output_to_input_mapping(
    std::vector<ParallelDimMappingRecord> const &mapping);
std::unordered_map<int, int> input_to_output_mapping(
    std::vector<ParallelDimMappingRecord> const &mapping);

class NoOp;

ParallelConfig get_basic_data_parallel_config(int num_parts, int dims);

class Aggregate;
class AggregateSpec;
class BatchMatmul;
class Cast;
class Concat;
class Conv2D;
class Dropout;
class ElementBinary;
class ElementUnary;
class Embedding;
class Experts;
class Flat;
class Gather;
class Group_by;
class LayerNorm;
class ResidualLayerNorm;
class AddBiasResidualLayerNorm;
class SigmoidSiluMulti;
class Linear;
class MultiHeadAttention;
class IncMultiHeadSelfAttention;
class TreeIncMultiHeadSelfAttention;
class Pool2D;
class Reduce;
class Reshape;
class Softmax;
class Split;
class TopK;
class GumbelTopK;
class ArgTopK;
class Transpose;
class RMSNorm;
class ResidualRMSNorm;
// class BeamTopK;
class SpecIncMultiHeadSelfAttention;
class Sampling;
class ArgMax;
class Combine;
class Repartition;
class Reduction;
class Replicate;
class AllReduce;
class FusedParallelOp;
class ParallelOpInfo;

// TODO: Move to an appropriate place
/*
  This is used to create a type that recursively replaces value type
  ParallelTensor by ParallelTensorShape in T. E.g., ToShape<std::tuple<int,
  ParallelTensor>>::type gives std::tuple<int, ParallelTensorShape>
*/
template <typename T>
struct ToShape {
  using type = T;
};

template <>
struct ToShape<ParallelTensor> {
  using type = ParallelTensorShape;
};

template <typename... Args, template <typename...> class Container>
struct ToShape<Container<Args...>> {
  using type = Container<typename ToShape<Args>::type...>;
};

// TODO: Move to an appropriate place
template <typename Input>
typename ToShape<Input>::type get_input_shape(Input const &input) = delete;

template <>
std::tuple<> get_input_shape(std::tuple<> const &);

template <>
std::tuple<ParallelTensorShape, ParallelTensorShape, ParallelTensorShape>
    get_input_shape(
        std::tuple<ParallelTensor, ParallelTensor, ParallelTensor> const &);

template <>
ParallelTensorShape get_input_shape(ParallelTensor const &input);

template <>
std::pair<ParallelTensorShape, ParallelTensorShape>
    get_input_shape(std::pair<ParallelTensor, ParallelTensor> const &inputs);

template <>
std::vector<ParallelTensorShape>
    get_input_shape(std::vector<ParallelTensor> const &inputs);

class FFModel {
public:
  FFModel(FFConfig &config, bool cpu_offload = false);
  ~FFModel();

  static constexpr float PROPAGATION_CHANCE = 0.25;
  static constexpr float CONTINUE_PROPAGATION_CHANCE = 0.75;
  static constexpr float PROPAGATION_SIZE_WEIGHT = 1.0;

  bool cpu_offload;
  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(Tensor const x, char const *name = NULL);
  // Add an add layer
  Tensor add(Tensor const x,
             Tensor const y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a subtract layer
  Tensor subtract(Tensor const x,
                  Tensor const y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a multiply layer
  Tensor multiply(Tensor const x,
                  Tensor const y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a divide layer
  Tensor divide(Tensor const x,
                Tensor const y,
                bool inplace_a = false,
                char const *name = NULL);
  // Add a max layer
  Tensor max(Tensor const x,
             Tensor const y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a min layer
  Tensor min(Tensor const x,
             Tensor const y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a rsqrt layer
  Tensor rsqrt(Tensor const x, bool inplace = true, char const *name = NULL);
  // Add a pow layer
  Tensor pow(Tensor const x,
             float const exponent,
             bool inplace = true,
             char const *name = NULL);
  // Add a scalar multiply layer
  Tensor scalar_multiply(Tensor const x,
                         float const scalar,
                         bool inplace = true,
                         char const *name = NULL);
  Tensor scalar_add(Tensor const x,
                    float const scalar,
                    bool inplace = true,
                    char const *name = NULL);
  Tensor scalar_sub(Tensor const x,
                    float const scalar,
                    bool inplace = true,
                    char const *name = NULL);
  Tensor scalar_truediv(Tensor const x,
                        float const scalar,
                        bool inplace = true,
                        char const *name = NULL);
  // Add a sin layer
  Tensor sin(Tensor const x, char const *name = NULL);
  // Add a cos layer
  Tensor cos(Tensor const x, char const *name = NULL);
  // Add an activation layer
  Tensor relu(Tensor const x, bool inplace = true, char const *name = NULL);
  Tensor identity(Tensor const x, char const *name = NULL);
  Tensor gelu(Tensor const x, char const *name = NULL);
  Tensor sigmoid(Tensor const x, char const *name = NULL);
  Tensor tanh(Tensor const x, char const *name = NULL);
  Tensor elu(Tensor const x, bool inplace = true, char const *name = NULL);
  // Add a 2D convolutional layer
  Tensor conv2d(Tensor const input,
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
  Tensor dropout(Tensor const input,
                 float rate,
                 unsigned long long seed = 0,
                 char const *name = NULL);
  // Add an embedding layer
  Tensor embedding(Tensor const input,
                   int num_entries,
                   int outDim,
                   AggrMode aggr,
                   DataType dtype = DT_FLOAT,
                   Layer const *shared_op = NULL,
                   Initializer *kernel_initializer = NULL,
                   char const *name = NULL);
  // Add a gather layer
  Tensor gather(Tensor const input,
                Tensor const index,
                int dim,
                char const *name = NULL);
  // Add a group_by layer
  void group_by(Tensor const data,
                Tensor const assign,
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
  Tensor pool2d(Tensor const input,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE,
                char const *name = NULL);
  // Add a layer_norm layer
  Tensor layer_norm(Tensor const input,
                    std::vector<int> const &axes,
                    bool elementwise_affine,
                    float eps,
                    bool use_bias = true,
                    DataType data_type = DT_NONE,
                    char const *name = NULL);
  // Add a layer_norm layer with residual(s)
  void residual_layer_norm(Tensor const input,
                           Tensor const residual1,
                           Tensor const residual2,
                           Tensor *outputs,
                           bool use_two_residuals,
                           std::vector<int> const &axes,
                           bool elementwise_affine,
                           float eps,
                           bool use_bias = true,
                           DataType data_type = DT_NONE,
                           char const *name = NULL);
  // Add a add_bias_residual_layer_norm layer
  void add_bias_residual_layer_norm(Tensor const input,
                                    Tensor const residual,
                                    Tensor *outputs,
                                    std::vector<int> const &axes,
                                    bool elementwise_affine,
                                    float eps,
                                    bool use_bias = true,
                                    DataType data_type = DT_NONE,
                                    char const *name = NULL);
  // Add a sigmoid_silu_multi layer
  Tensor sigmoid_silu_multi(Tensor const input1,
                            Tensor const input2,
                            DataType data_type = DT_NONE,
                            char const *name = NULL);
  // Add a batch_norm layer
  Tensor
      batch_norm(Tensor const input, bool relu = true, char const *name = NULL);
  // Add a batch_matmul layer
  Tensor batch_matmul(Tensor const A,
                      Tensor const B,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      char const *name = nullptr);
  // Add a root mean square layer
  Tensor rms_norm(Tensor const input,
                  float eps,
                  int dim,
                  DataType data_type = DT_NONE,
                  char const *name = NULL);
  // Add a residual root mean square layer
  void residual_rms_norm(Tensor const input1,
                         Tensor const input2,
                         Tensor *outputs,
                         float eps,
                         int dim,
                         DataType data_type = DT_NONE,
                         char const *name = NULL);
  //   // Add a beam search top k layer
  //   Tensor beam_top_k(Tensor const input,
  //                     int max_beam_size,
  //                     bool sorted,
  //                     char const *name = NULL);

  // Add a dense layer
  Tensor dense(Tensor const input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               DataType data_type = DT_NONE,
               Layer const *shared_op = NULL,
               Initializer *kernel_initializer = NULL,
               Initializer *bias_initializer = NULL,
               RegularizerMode regularizer_type = REG_MODE_NONE,
               float regularizer_lambda = 0.0,
               char const *name = NULL);
  // Add a cast layer
  Tensor cast(Tensor const input, DataType dtype, char const *name = nullptr);
  // Add a concat layer
  Tensor
      concat(int n, Tensor const *tensors, int axis, char const *name = NULL);
  // Add an experts layer
  Tensor experts(
      Tensor const *inputs,
      int num_experts,
      int experts_start_idx,
      int experts_output_dim_size,
      float alpha,
      int experts_num_layers = 1,        // number of linear layers per expert
      int experts_internal_dim_size = 0, // hidden dimension for internal layers
      char const *name = NULL);
  // Add a mean layer
  Tensor mean(Tensor const input,
              std::vector<int> const &dims,
              bool keepdims,
              char const *name);
  // Add a moe layer (wrapping topk, group_by and aggregate operators)
  Tensor moe(Tensor const input,
             int num_exp,
             int num_select,
             int expert_hidden_size,
             float alpha,
             float lambda);
  // Add a split layer
  void split(Tensor const input,
             Tensor *outputs,
             std::vector<int> const &split,
             int axis,
             char const *name = NULL);
  // Add a flat layer
  Tensor flat(Tensor const input, char const *name = NULL);
  // Add a softmax layer
  Tensor softmax(Tensor const input,
                 int dim = -1,
                 DataType data_type = DT_NONE,
                 char const *name = NULL);
  // Create input tensors and constants
  Tensor transpose(Tensor const input,
                   std::vector<int> const &perm,
                   char const *name = NULL);
  Tensor reduce_sum(Tensor const input,
                    std::vector<int> const &axes,
                    bool keepdims = false,
                    char const *name = nullptr);
  Tensor reshape(Tensor const input,
                 std::vector<int> const &shape,
                 char const *name = NULL);
  Tensor reverse(Tensor const input, int axis, char const *name = NULL);
  void top_k(Tensor const input,
             Tensor *outputs,
             int k,
             bool sorted,
             char const *name = NULL);
  Tensor gumbel_top_k(Tensor const input,
                      // Tensor *outputs,
                      int k,
                      bool sorted,
                      bool speculative_decoding,
                      char const *name = NULL);
  Tensor arg_top_k(Tensor const input,
                   // Tensor *outputs,
                   int k,
                   bool sorted,
                   bool renormalize,
                   char const *name = NULL);
  Tensor argmax(Tensor const input, bool beam_search, char const *name = NULL);
  Tensor sampling(Tensor const input, float top_p, char const *name = NULL);
  Tensor multihead_attention(Tensor const query,
                             Tensor const key,
                             Tensor const value,
                             int embed_dim,
                             int num_heads,
                             int kdim = 0,
                             int vdim = 0,
                             float dropout = 0.0f,
                             bool bias = true,
                             bool add_bias_kv = false,
                             bool add_zero_attn = false,
                             DataType data_type = DT_NONE,
                             Initializer *kernel_initializer = NULL,
                             char const *name = NULL);
  Tensor inc_multihead_self_attention(Tensor const input,
                                      int embed_dim,
                                      int num_heads,
                                      int kdim = 0,
                                      int vdim = 0,
                                      float dropout = 0.0f,
                                      bool bias = false,
                                      bool add_bias_kv = false,
                                      bool add_zero_attn = false,
                                      DataType data_type = DT_NONE,
                                      Initializer *kernel_initializer = NULL,
                                      bool apply_rotary_embedding = false,
                                      bool scaling_query = false,
                                      float scaling_factor = 1.0f,
                                      bool qk_prod_scaling = true,
                                      bool position_bias = false,
                                      bool streaming_cache = false,
                                      char const *name = NULL);
  Tensor
      spec_inc_multihead_self_attention(Tensor const input,
                                        int embed_dim,
                                        int num_heads,
                                        int kdim = 0,
                                        int vdim = 0,
                                        float dropout = 0.0f,
                                        bool bias = false,
                                        bool add_bias_kv = false,
                                        bool add_zero_attn = false,
                                        DataType data_type = DT_NONE,
                                        Initializer *kernel_initializer = NULL,
                                        bool apply_rotary_embedding = false,
                                        bool scaling_query = false,
                                        float scaling_factor = 1.0f,
                                        bool qk_prod_scaling = true,
                                        bool position_bias = false,
                                        bool streaming_cache = false,
                                        char const *name = NULL);
  Tensor inc_multihead_self_attention_verify(
      Tensor const input,
      int embed_dim,
      int num_heads,
      int kdim = 0,
      int vdim = 0,
      float dropout = 0.0f,
      bool bias = false,
      bool add_bias_kv = false,
      bool add_zero_attn = false,
      DataType data_type = DT_NONE,
      Initializer *kernel_initializer = NULL,
      bool apply_rotary_embedding = false,
      bool scaling_query = false,
      float scaling_factor = 1.0f,
      bool qk_prod_scaling = true,
      bool position_bias = false,
      char const *name = NULL);
  Tensor groupquery_self_attention(Tensor const input,
                                   int embed_dim,
                                   int num_q_heads,
                                   int num_kv_heads,
                                   int kdim = 0,
                                   int vdim = 0,
                                   float dropout = 0.0f,
                                   bool bias = false,
                                   bool add_bias_kv = false,
                                   bool add_zero_attn = false,
                                   DataType data_type = DT_NONE,
                                   Initializer *kernel_initializer = NULL,
                                   bool apply_rotary_embedding = false,
                                   bool scaling_query = false,
                                   float scaling_factor = 1.0f,
                                   bool qk_prod_scaling = true,
                                   bool position_bias = false,
                                   bool streaming_cache = false,
                                   char const *name = NULL);
  Tensor
      spec_inc_multiquery_self_attention(Tensor const input,
                                         int embed_dim,
                                         int num_q_heads,
                                         int num_kv_heads,
                                         int kdim = 0,
                                         int vdim = 0,
                                         float dropout = 0.0f,
                                         bool bias = false,
                                         bool add_bias_kv = false,
                                         bool add_zero_attn = false,
                                         DataType data_type = DT_NONE,
                                         Initializer *kernel_initializer = NULL,
                                         bool apply_rotary_embedding = false,
                                         bool scaling_query = false,
                                         float scaling_factor = 1.0f,
                                         bool qk_prod_scaling = true,
                                         bool position_bias = false,
                                         bool streaming_cache = false,
                                         char const *name = NULL);
  Tensor inc_multiquery_self_attention_verify(
      Tensor const input,
      int embed_dim,
      int num_q_heads,
      int num_kv_heads,
      int kdim = 0,
      int vdim = 0,
      float dropout = 0.0f,
      bool bias = false,
      bool add_bias_kv = false,
      bool add_zero_attn = false,
      DataType data_type = DT_NONE,
      Initializer *kernel_initializer = NULL,
      bool apply_rotary_embedding = false,
      bool scaling_query = false,
      float scaling_factor = 1.0f,
      bool qk_prod_scaling = true,
      bool position_bias = false,
      char const *name = NULL);
  // ========================================
  // Inference APIs
  // ========================================
  std::vector<GenerationResult> generate(std::vector<std::string> &prompts,
                                         EmissionMachine &emission_machine);

  std::vector<GenerationResult>
      generate(std::vector<GenerationRequest> &requests,
               EmissionMachine &emission_machine);

  Tensor create_tensor_legion_ordering(int num_dim,
                                       int const dims[],
                                       DataType data_type,
                                       Layer const *owner_op = NULL,
                                       int owner_idx = 0,
                                       bool create_grad = true);
  ParallelTensor
      create_parallel_tensor_legion_ordering(int num_dim,
                                             ParallelDim const dims[],
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
                                        ParallelDim const dims[],
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
  ParallelTensor create_parallel_tensor(ParallelDim const dims[],
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
      ParallelDim const dims[],
      DataType data_type,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  ParallelParameter create_parallel_weight(
      int numdim,
      ParallelDim const dims[],
      DataType data_type,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  ParallelParameter create_parallel_weight_legion_ordering(
      int numdim,
      ParallelDim const dims[],
      DataType data_type,
      Op const *owner_op = NULL,
      bool create_grad = true,
      Initializer *initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);

  void map_tensor(ParallelTensor tensor, Op const *parallel_op);
  void map_weight(ParallelTensor tensor, Op const *parallel_op);
  bool get_parallel_tensor_from_tensor(Tensor const tensor,
                                       ParallelTensor &parallel_tensor) const;

  template <int NDIM>
  Tensor create_constant(int const dims[], float value, DataType date_type);
  // ========================================
  // Graph APIs
  // ========================================
  float graph_cost(const PCG::Graph *graph,
                   const PCG::Node &sink_node,
                   MachineView const &sink_view,
                   const PCG::Node &source_node,
                   MachineView const &source_view,
                   MachineResource const &resources,
                   bool include_sink_compute_time,
                   bool constructing_optimal_view = false);
  void construct_optimal_view(
      const PCG::Graph *graph,
      const PCG::Node &sink_node,
      MachineView const &sink_view,
      const PCG::Node &source_node,
      MachineView const &source_view,
      MachineResource const &resources,
      bool include_sink_compute_time,
      float optimal_cost,
      std::unordered_map<PCG::Node, MachineView> &optimal_views);
  void deserialize_graph_optimal_view(
      Legion::Deserializer &dez,
      PCG::Graph *graph,
      std::unordered_map<PCG::Node, MachineView> &optimal_views);
  bool convert_graph_to_operators(
      const PCG::Graph *graph,
      std::unordered_map<PCG::Node, MachineView> const &optimal_views);
  static void register_all_machine_views(int num_nodes,
                                         int gpus_per_node,
                                         int cpus_per_node,
                                         std::vector<MachineView> &valid_views);
  // ========================================
  // Internal PCG::Node creation APIs
  // ========================================
  template <typename T>
  PCG::Node get_or_create_node(typename T::Input const &input,
                               typename T::Params const &params) {
    using Params = typename T::Params;

    auto input_shapes = get_input_shape<typename T::Input>(input);

    if (!params.is_valid(input_shapes)) {
      printf("!params.is_valid(input_shapes)\n");
      return PCG::Node::INVALID_NODE;
    }

    T *op = nullptr;

    std::pair<typename ToShape<typename T::Input>::type, Params> key{
        input_shapes, params};
    auto &cache = FlexFlow::get<std::unordered_map<
        std::pair<typename ToShape<typename T::Input>::type, Params>,
        T *>>(this->cached_ops);
    auto const &it = cache.find(key);
    if (it != cache.end()) {
      op = it->second;
    } else {
      op = new T(*this, params, input);
      cache[key] = op;
    }

    assert(op->get_params() == params);
    return this->new_node(op);
  }

  PCG::Node get_or_create_noop_node(ParallelTensor const input);
  PCG::Node get_or_create_input_node(ParallelTensorShape const &);
  PCG::Node get_or_create_fused_parallel_node(
      ParallelTensor const input,
      std::vector<ParallelOpInfo> const &parallel_ops);
  PCG::Node get_or_create_parallel_op_node(ParallelTensor const input,
                                           ParallelOpInfo const &);
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  void create_disjoint_partition(int num_dims,
                                 ParallelDim const dims[],
                                 Legion::IndexSpace const &part_is,
                                 Legion::LogicalRegion const &region,
                                 Legion::LogicalPartition &part);
  template <int NDIM, int TDIM>
  void create_disjoint_partition_with_dim2(
      ParallelDim const dims[],
      Legion::IndexSpaceT<TDIM> const &part_is,
      Legion::LogicalRegion const &region,
      Legion::LogicalPartition &part);
  void create_aliased_partition(int num_dims,
                                ParallelDim const dims[],
                                int aliased_dim,
                                Legion::IndexSpace const &part_is,
                                Legion::LogicalRegion const &region,
                                Legion::LogicalPartition &part);
  template <int NDIM, int TDIM>
  void create_aliased_partition_with_dim2(
      ParallelDim const dims[],
      int aliased_dim,
      Legion::IndexSpaceT<TDIM> const &part_is,
      Legion::LogicalRegion const &region,
      Legion::LogicalPartition &part);

  template <int NDIM>
  void create_disjoint_partition(ParallelTensor const tensor,
                                 Legion::IndexSpaceT<NDIM> const &part_is,
                                 Legion::LogicalPartition &part_fwd,
                                 Legion::LogicalPartition &part_bwd);

  template <int NDIM, int TDIM>
  void create_data_parallel_partition_with_diff_dims(
      ParallelTensor const tensor,
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
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  void reset_metrics();
  void init_operators();
  void init_operators_inference(
      std::vector<ParallelTensor> const &batch_inputs,
      std::vector<ParallelTensor> const &batch_outputs);
  void prefetch();
  void forward(int seq_length = -1);
  void compute_metrics();
  void get_metrics();
  void backward(int seq_length = -1);
  void update();
  bool apply_fusion(
      std::vector<Op *> const &operators,
      std::vector<Op *> &new_operators,
      std::unordered_map<ParallelTensor, std::vector<ParallelTensor>>
          *parallel_tensor_mapping = nullptr);
  bool check_operators_integrity(
      std::vector<Op *> const &old_operators,
      std::unordered_map<ParallelTensor, std::vector<ParallelTensor>>
          *pt_mapping = nullptr);
  Op *get_final_operator() const;
  void compile(LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void compile(Optimizer *optimizer,
               LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void compile_inference();
  void set_transformer_layer_id(int id);
  void set_position_offset(int offset);
  void graph_optimize(size_t budget,
                      bool only_data_parallel,
                      std::unique_ptr<PCG::Graph> &best_graph,
                      std::unordered_map<PCG::Node, MachineView> &optimal_view);
  void graph_optimize(size_t budget,
                      bool only_data_parallel,
                      std::unique_ptr<PCG::Graph> &best_graph,
                      std::unordered_map<PCG::Node, MachineView> &optimal_view,
                      bool perform_memory_search,
                      MemoryOptimConfig new_config,
                      MemorySearchResult &search_result);
  void mcmc_optimize(std::map<Op const *, ParallelConfig> &best,
                     size_t budget,
                     float alpha,
                     CompMode comp_mode,
                     bool use_propagation) const;
#ifdef FF_USE_NCCL
  ncclComm_t *find_nccl_comms(MachineView const &view) const;
#endif
#ifdef FF_USE_PROPAGATE
  void propagate(std::map<Op *, ParallelConfig> const &current,
                 std::map<Op *, ParallelConfig> &next) const;
#endif
  void rewrite(std::map<Op const *, ParallelConfig> const &current,
               std::map<Op const *, ParallelConfig> &next,
               bool use_propagation) const;
  void recompile_on_condition(RecompileState &r);
  void zero_gradients();
  void print_layers(int id);

  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>>
      get_bwd_edge_map() const;

  // Internal functions
  Legion::IndexSpace get_or_create_task_is(ParallelConfig const &pc);
  Legion::IndexSpace get_or_create_task_is(MachineView const &view);
  Legion::IndexSpace get_or_create_task_is(Legion::Domain const &domain);
  Legion::IndexSpace get_or_create_task_is(ParallelTensor const);
  Legion::IndexSpace get_task_is(Legion::Domain const &domain) const;
  Legion::IndexSpace get_task_is(ParallelConfig const &pc) const;
  Legion::IndexSpace get_task_is(MachineView const &view) const;
  bool is_mlp_block(int layer_idx) const;
  void create_operators_from_layers();
  Op *create_operator_from_layer(Layer *layer,
                                 std::vector<ParallelTensor> const &inputs);
  // APIs for setting iteration configs
public:
  void set_iteration_config_sequence_length(int seq_length);

  /**
   * @brief Clear the cache of the GraphSearchHelper and SearchHelper.
   */
  void clear_graph_search_cache();

public:
  size_t op_global_guid, layer_global_guid;
  size_t tensor_global_guid, parallel_tensor_global_guid, node_global_guid;
  size_t current_transformer_layer_id;
  // positional embedding start offset
  int position_offset;
  FFConfig config;
  FFIterationConfig iter_config;
  Optimizer *optimizer;
  PCG::SearchHelper *search;
  PCG::GraphSearchHelper *graph_search;
  Loss *loss_op;
  Metrics *metrics_op;
  Simulator *simulator;
  int metrics_input;
  ParallelTensor parallel_label_tensor;
  Tensor label_tensor;

  std::vector<Layer *> layers;
  std::vector<Op *> operators;
  std::vector<ParallelTensor> parameters;
  FFHandler handlers[MAX_NUM_WORKERS];
  Legion::Future current_metrics;
  // Cached operators: key: operator hash, value: operator pointer
  std::tuple<
      std::unordered_map<
          std::pair<std::vector<ParallelTensorShape>, AggregateParams>,
          Aggregate *>,
      std::unordered_map<std::pair<ParallelTensorShape, AggregateSpecParams>,
                         AggregateSpec *>,
      std::unordered_map<
          std::pair<std::pair<ParallelTensorShape, ParallelTensorShape>,
                    BatchMatmulParams>,
          BatchMatmul *>,
      std::unordered_map<std::pair<ParallelTensorShape, CastParams>, Cast *>,
      std::unordered_map<
          std::pair<std::vector<ParallelTensorShape>, ConcatParams>,
          Concat *>,
      std::unordered_map<std::pair<ParallelTensorShape, Conv2DParams>,
                         Conv2D *>,
      std::unordered_map<std::pair<ParallelTensorShape, DropoutParams>,
                         Dropout *>,
      std::unordered_map<
          std::pair<std::pair<ParallelTensorShape, ParallelTensorShape>,
                    ElementBinaryParams>,
          ElementBinary *>,
      std::unordered_map<std::pair<ParallelTensorShape, ElementUnaryParams>,
                         ElementUnary *>,
      std::unordered_map<std::pair<ParallelTensorShape, EmbeddingParams>,
                         Embedding *>,
      std::unordered_map<
          std::pair<std::vector<ParallelTensorShape>, ExpertsParams>,
          Experts *>,
      std::unordered_map<std::pair<ParallelTensorShape, FlatParams>, Flat *>,
      std::unordered_map<
          std::pair<std::pair<ParallelTensorShape, ParallelTensorShape>,
                    GatherParams>,
          Gather *>,
      std::unordered_map<
          std::pair<std::pair<ParallelTensorShape, ParallelTensorShape>,
                    Group_byParams>,
          Group_by *>,
      std::unordered_map<std::pair<ParallelTensorShape, LayerNormParams>,
                         LayerNorm *>,
      std::unordered_map<std::pair<std::tuple<ParallelTensorShape,
                                              ParallelTensorShape,
                                              ParallelTensorShape>,
                                   ResidualLayerNormParams>,
                         ResidualLayerNorm *>,
      std::unordered_map<
          std::pair<std::pair<ParallelTensorShape, ParallelTensorShape>,
                    AddBiasResidualLayerNormParams>,
          AddBiasResidualLayerNorm *>,
      std::unordered_map<
          std::pair<std::pair<ParallelTensorShape, ParallelTensorShape>,
                    SigmoidSiluMultiParams>,
          SigmoidSiluMulti *>,
      std::unordered_map<std::pair<ParallelTensorShape, LinearParams>,
                         Linear *>,
      std::unordered_map<std::pair<ParallelTensorShape, Pool2DParams>,
                         Pool2D *>,
      std::unordered_map<std::pair<std::tuple<ParallelTensorShape,
                                              ParallelTensorShape,
                                              ParallelTensorShape>,
                                   MultiHeadAttentionParams>,
                         MultiHeadAttention *>,
      std::unordered_map<
          std::pair<ParallelTensorShape, IncMultiHeadSelfAttentionParams>,
          IncMultiHeadSelfAttention *>,
      //   std::unordered_map<std::pair<ParallelTensorShape, BeamTopKParams>,
      //                      BeamTopK *>,
      std::unordered_map<std::pair<ParallelTensorShape, SamplingParams>,
                         Sampling *>,
      std::unordered_map<std::pair<ParallelTensorShape, ArgMaxParams>,
                         ArgMax *>,
      std::unordered_map<
          std::pair<ParallelTensorShape, SpecIncMultiHeadSelfAttentionParams>,
          SpecIncMultiHeadSelfAttention *>,
      std::unordered_map<
          std::pair<ParallelTensorShape, TreeIncMultiHeadSelfAttentionParams>,
          TreeIncMultiHeadSelfAttention *>,
      std::unordered_map<std::pair<ParallelTensorShape, ReduceParams>,
                         Reduce *>,
      std::unordered_map<std::pair<ParallelTensorShape, ReshapeParams>,
                         Reshape *>,
      std::unordered_map<std::pair<ParallelTensorShape, SplitParams>, Split *>,
      std::unordered_map<std::pair<ParallelTensorShape, SoftmaxParams>,
                         Softmax *>,
      std::unordered_map<std::pair<ParallelTensorShape, TopKParams>, TopK *>,
      std::unordered_map<std::pair<ParallelTensorShape, GumbelTopKParams>,
                         GumbelTopK *>,
      std::unordered_map<std::pair<ParallelTensorShape, ArgTopKParams>,
                         ArgTopK *>,
      std::unordered_map<std::pair<ParallelTensorShape, TransposeParams>,
                         Transpose *>,
      std::unordered_map<std::pair<ParallelTensorShape, RMSNormParams>,
                         RMSNorm *>,
      std::unordered_map<
          std::pair<std::pair<ParallelTensorShape, ParallelTensorShape>,
                    ResidualRMSNormParams>,
          ResidualRMSNorm *>,
      std::unordered_map<std::pair<ParallelTensorShape, RepartitionParams>,
                         Repartition *>,
      std::unordered_map<std::pair<ParallelTensorShape, ReplicateParams>,
                         Replicate *>,
      std::unordered_map<std::pair<ParallelTensorShape, ReductionParams>,
                         Reduction *>,
      std::unordered_map<std::pair<ParallelTensorShape, CombineParams>,
                         Combine *>,
      std::unordered_map<std::pair<ParallelTensorShape, AllReduceParams>,
                         AllReduce *>,
      std::unordered_map<std::pair<ParallelTensorShape, FusedParallelOpParams>,
                         FusedParallelOp *>>
      cached_ops;
  std::unordered_map<size_t, NoOp *> cached_noop_ops;
  std::unordered_map<size_t, NoOp *> cached_input_ops;
  std::vector<MachineView> all_valid_views;
  int model_id; // unique incremental id assigned to each model. Used in the
                // inference_debugging mode.
#ifdef FF_USE_NCCL
  std::unordered_map<size_t, ncclComm_t *> view_hash_to_nccl_comms;
#endif
private:
  bool debug;
  std::map<MachineView, Legion::IndexSpace, MachineViewDimCompare> all_task_is;

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
  PCG::Node new_node(Op *);
  static int model_counter; // number of instantiated FFModel objects. Used to
                            // assign a unique incremental id to each model.
                            // Used in the inference_debugging mode.
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

void register_flexflow_internal_tasks(Legion::Runtime *runtime = NULL,
                                      bool pre_register = true,
                                      bool enable_control_replication = true);

void register_custom_tasks();

}; // namespace FlexFlow

#endif //_FLEXFLOW_MODEL_H_

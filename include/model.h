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
#ifndef _FLEXFLOW_MODEL_H_
#define _FLEXFLOW_MODEL_H_
#include "legion.h"
#include "config.h"
#include "tensor.h"
#include "initializer.h"
#include "simulator.h"
#include "optimizer.h"
#include "accessor.h"
#include "loss_functions.h"
#include "metrics_functions.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <unistd.h>
#include <functional>
#include "tl/optional.h"

#include "ffconst.h"

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FF_INIT_TASK_ID,
  IMAGE_INIT_TASK_ID,
  LABEL_INIT_TASK_ID,
  LOAD_IMAGES_TASK_ID,
  NORMALIZE_IMAGES_TASK_ID,
  ELEMENTBINARY_INIT_TASK_ID,
  ELEMENTBINARY_FWD_TASK_ID,
  ELEMENTBINARY_BWD_TASK_ID,
  ELEMENTUNARY_INIT_TASK_ID,
  ELEMENTUNARY_FWD_TASK_ID,
  ELEMENTUNARY_BWD_TASK_ID,
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
  EMBED_BWD_TASK_ID,
  GROUP_BY_INIT_TASK_ID,
  GROUP_BY_FWD_TASK_ID,
  GROUP_BY_BWD_TASK_ID,
  AGGREGATE_INIT_TASK_ID,
  AGGREGATE_FWD_TASK_ID,
  AGGREGATE_BWD_TASK_ID,
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
  LINEAR_INIT_TASK_ID,
  LINEAR_INIT_PARA_TASK_ID,
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
  CONCAT_INIT_TASK_ID,
  CONCAT_FWD_TASK_ID,
  CONCAT_BWD_TASK_ID,
  SPLIT_INIT_TASK_ID,
  SPLIT_FWD_TASK_ID,
  SPLIT_BWD_TASK_ID,
  RESHAPE_INIT_TASK_ID,
  RESHAPE_FWD_TASK_ID,
  RESHAPE_BWD_TASK_ID,
  REVERSE_INIT_TASK_ID,
  REVERSE_FWD_TASK_ID,
  REVERSE_BWD_TASK_ID,
  TOPK_INIT_TASK_ID,
  TOPK_FWD_TASK_ID,
  TOPK_BWD_TASK_ID,
  TRANSPOSE_INIT_TASK_ID,
  TRANSPOSE_FWD_TASK_ID,
  TRANSPOSE_BWD_TASK_ID,
  ATTENTION_INIT_TASK_ID,
  ATTENTION_FWD_TASK_ID,
  ATTENTION_BWD_TASK_ID,
  MSELOSS_BWD_TASK_ID,
  FUSEDOP_INIT_TASK_ID,
  FUSEDOP_FWD_TASK_ID,
  FUSEDOP_BWD_TASK_ID,
  NOOP_INIT_TASK_ID,
  //Metrics tasks
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
  // Search
  STRATEGY_SEARCH_TASK_ID,
  // Graph
  GRAPH_OPTIMIZE_TASK_ID,
  // Python data loader
  PY_DL_FLOAT_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_FLOAT_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_FLOAT_LOAD_BATCH_GPU_TASK_ID,
  PY_DL_INT_LOAD_BATCH_GPU_TASK_ID,
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
  FUSED_PARALLELOP_INIT_TASK_ID,
  FUSED_PARALLELOP_FWD_TASK_ID,
  FUSED_PARALLELOP_BWD_TASK_ID,
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
};

enum ShardingID {
  DataParallelShardingID = 135,
};

enum FieldIDs {
  FID_DATA,
};

namespace Legion {
  class Serializer;
}

class SearchHelper;
class FFModel;
class Op;
class DataLoader;

class OpMeta {
public:
  OpMeta(FFHandler _handle);
public:
  FFHandler handle;
  bool profiling; // Measure the run time of the task
};

enum class MappingRecordType {
  INPUT_OUTPUT,
  INPUT_WEIGHT
};

enum class MappingOperation {
  PARTITION,
  REPLICATE
};

class ParallelDimMappingRecord {
private:
  ParallelDimMappingRecord(MappingRecordType);
public:
  ParallelDimMappingRecord() = delete;

  static ParallelDimMappingRecord input_output_record(int input_idx, int input_dim, 
                                                      int output_idx, int output_dim,
                                                      tl::optional<MappingOperation> operation = tl::nullopt);
  static ParallelDimMappingRecord input_weight_record(int input_idx, int input_dim,
                                                      int weight_idx, int weight_dim,
                                                      tl::optional<MappingOperation> operation = tl::nullopt);
  MappingRecordType get_type() const;
public:
  MappingRecordType type;
  tl::optional<MappingOperation> operation;

  int output_dim, input_dim, weight_dim;
  int output_idx, input_idx, weight_idx;
};

std::string optype_to_string(OperatorType);

void solve_parallel_dim_mappings(const std::vector<ParallelDimMappingRecord>& mapping,
                                 const std::vector<ParallelDim const *> &inputs,
                                 const std::vector<ParallelDim *> &weights, 
                                 const std::vector<ParallelDim *> &outputs);
std::unordered_map<int, int> output_to_input_mapping(const std::vector<ParallelDimMappingRecord>& mapping);
std::unordered_map<int, int> input_to_output_mapping(const std::vector<ParallelDimMappingRecord>& mapping);

class Op {
public:
  static void construct_weight_parallel_dims(
      std::vector<ParallelDimMappingRecord>& records, 
      std::vector<std::pair<int, int>> mappings, int input_idx = 0, int weight_idx = 0);
  static void construct_weight_parallel_dims(
      std::vector<ParallelDimMappingRecord>& records, 
      std::vector<std::tuple<int, MappingOperation, int>> mappings, int input_idx = 0, int weight_idx = 0);
  static void construct_weight_parallel_dims(
      std::vector<ParallelDimMappingRecord>& records,
      int input_dim, int weight_dim, 
      int input_idx = 0, int weight_idx = 0, 
      tl::optional<MappingOperation> operation = tl::nullopt);

  static void construct_output_parallel_dims(
      std::vector<ParallelDimMappingRecord>& records, 
      std::vector<std::pair<int, int>> mappings, int input_idx = 0, int output_idx = 0);
  static void construct_output_parallel_dims(
      std::vector<ParallelDimMappingRecord>& records, 
      std::vector<std::tuple<int, MappingOperation, int>> mappings, int input_idx = 0, int output_idx = 0);
  static void construct_output_parallel_dims(
      std::vector<ParallelDimMappingRecord>& records,
      int input_dim, int output_dim, 
      int input_idx = 0, int output_idx = 0, 
      tl::optional<MappingOperation> operation = tl::nullopt);
protected:
  void register_weight_parallel_dims(
      std::vector<std::pair<int, int>> mappings, int input_idx = 0, int weight_idx = 0);
  void register_weight_parallel_dims(
      std::vector<std::tuple<int, MappingOperation, int>> mappings, int input_idx = 0, int weight_idx = 0);
  void register_weight_parallel_dims(
      int input_dim, int weight_dim, 
      int input_idx = 0, int weight_idx = 0, 
      tl::optional<MappingOperation> operation = tl::nullopt);

  void register_output_parallel_dims(
      std::vector<std::pair<int, int>> mappings, int input_idx = 0, int output_idx = 0);
  void register_output_parallel_dims(
      std::vector<std::tuple<int, MappingOperation, int>> mappings, int input_idx = 0, int output_idx = 0);
  void register_output_parallel_dims(
      int input_dim, int output_dim, 
      int input_idx = 0, int output_idx = 0, 
      tl::optional<MappingOperation> operation = tl::nullopt);

  int get_output_to_input_dim_mapping(
      const Tensor output, int output_dim,
      const Tensor input);
  int get_output_to_weight_dim_mapping(
      const Tensor output, int output_dim,
      const Tensor weight);

  void inner_measure_operator_cost(Simulator *sim,
                                   std::function<void()> const &forward,
                                   std::function<void()> const &backward,
                                   CostMetrics& cost_metrics) const;

  bool check_output_input_weight_parallel_dims(bool allocate_weights = true) const;
  bool check_output_input_weight_same_parallel_is() const;
  bool check_output_input_weight_same_machine_view() const;
public:
  Op(FFModel& model,
     OperatorType type,
     const char* _name,
     int numInputs,
     int numWeights,
     bool allocate_weights,
     int numOutputs,
     const Tensor input1 = NULL,
     const Tensor input2 = NULL,
     const Tensor input3 = NULL,
     const Tensor input4 = NULL);
  Op(FFModel& model,
     OperatorType type,
     const char* _name,
     int numInputs,
     int numWeights,
     int numOutputs,
     const Tensor input1 = NULL,
     const Tensor input2 = NULL,
     const Tensor input3 = NULL,
     const Tensor input4 = NULL);
  Op(FFModel& model,
     OperatorType type,
     const char* _name,
     int numInputs,
     int numWeights,
     int numOutputs,
     const Tensor* tensors);
  // graph substitution related methods
  virtual bool get_int_parameter(PMParameter, int*) const;
  virtual bool get_tensor_parameter(TNParameter, DIMParameter, int*) const;
  virtual bool get_input_parameter(TNParameter, DIMParameter, int*) const;
  virtual bool get_weight_parameter(TNParameter, DIMParameter, int*) const;
  // Pure virtual functions that must be implemented
  virtual void init(const FFModel&) = 0;
  virtual void forward(const FFModel&) = 0;
  virtual void backward(const FFModel&) = 0;
  virtual void print_layer(const FFModel& model) = 0;
  virtual bool measure_operator_cost(Simulator* sim,
                                     const ParallelConfig& pc,
                                     CostMetrics& cost_metrics) const = 0;
  virtual bool estimate_sync_cost(Simulator* sim,
                                  const MachineView& pc,
                                  CostMetrics& cost_metrics) const;
  // Other virtual functions that can be optionally overwritten
  virtual ParallelConfig get_random_parallel_config(const FFModel& ff) const;
  virtual ParallelConfig get_data_parallel_config(const FFModel& ff) const;
  virtual Legion::Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx) const;
  virtual Legion::Domain get_output_tensor_shape(const ParallelConfig& pc, int output_idx, int part_idx) const;
  virtual Legion::Domain get_weight_tensor_shape(const ParallelConfig& pc, int weight_idx, int part_idx) const;
  virtual bool is_valid_parallel_config(const FFModel& ff, const ParallelConfig& pc) const;
  virtual bool is_adoptable_parallel_config(FFModel const &ff, ParallelConfig const &pc) const;
  // Helper functions
  void prefetch(const FFModel&);
  void zero_grad(const FFModel&);
  Tensor get_parameter(int index);
  virtual bool can_inplace_output();
  virtual bool has_inplace_output();
  virtual void do_inplace_output();
  virtual bool is_parallel_op() const;
  virtual void serialize(Legion::Serializer&) const;
  virtual Op *materialize(FFModel& ff, Tensor inputs[], int num_inputs) const;

  int get_dimension() const;
#ifdef FF_USE_NCCL
  static ncclUniqueId get_nccl_unique_id_task(const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx, Legion::Runtime *runtime);
  static ncclComm_t init_nccl_comms_task(const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx, Legion::Runtime *runtime);
#endif
protected:
  void set_argumentmap_for_init(const FFModel& ff, Legion::ArgumentMap& argmap);
  void set_argumentmap_for_forward(const FFModel& ff, Legion::ArgumentMap& argmap);
  void set_argumentmap_for_backward(const FFModel& ff, Legion::ArgumentMap& argmap);
  void set_opmeta_from_futuremap(const FFModel& ff, const Legion::FutureMap& fm);
  void solve_parallel_dim_mappings(const std::vector<ParallelDim const *> &inputs,
                                   const std::vector<ParallelDim *> &weights, 
                                   const std::vector<ParallelDim *> &outputs) const;
public:
  OperatorType op_type;
  size_t op_guid;
  char name[MAX_OPNAME];
  Legion::IndexSpace parallel_is;
  Tensor outputs[MAX_NUM_OUTPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  Tensor weights[MAX_NUM_WEIGHTS];
  OpMeta* meta[MAX_NUM_WORKERS];
  int numInputs, numWeights, numOutputs;
  bool profiling;
#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
#endif
  // Note: parallel_dims_mapping should not be called in a DNN task
  std::vector<ParallelDimMappingRecord>* parallel_dims_mapping;
};

struct Node {
  Node(void);
  Node(size_t _guid, Op* _ptr): guid(_guid), ptr(_ptr) {}
  inline bool operator==(const Node& b) const {
    if (guid != b.guid) return false;
    if (ptr != b.ptr) return false;
    return true;
  }
  inline bool operator!=(const Node& b) const {
    if (guid != b.guid) return true;
    if (ptr != b.ptr) return true;
    return false;
  }
  inline bool operator<(const Node& b) const {
    if (guid != b.guid) return guid < b.guid;
    if (ptr != b.ptr) return ptr < b.ptr;
    return false;
  }
  Node& operator=(const Node& n)
  {
    guid = n.guid;
    ptr = n.ptr;
    return *this;
  }
  std::string op_to_string(const Op* ptr) const;
  std::string to_string(void) const
  {
    if (ptr != NULL) {
      return op_to_string(ptr);
    }
    else {
      return "UnmappedOp_" + std::to_string(guid);
    }
  }
  static const Node INVALID_NODE;
  size_t guid;
  const Op* ptr;
};

class NoOp;

ParallelConfig get_basic_data_parallel_config(int num_parts, int dims);

class Concat;
class Conv2D;
class Conv2DParams;
class ElementBinary;
class ElementUnary;
class Embedding;
class Flat;
class Linear;
class MultiHeadAttention;
class Pool2D;
class Pool2DParams;
class Softmax;
class Combine;
class Repartition;
class Reduction;
class Replicate;
class FusedParallelOp;
class Graph;

class FFModel {
public:
  FFModel(FFConfig &config);

  static constexpr float PROPAGATION_CHANCE = 0.25;
  static constexpr float CONTINUE_PROPAGATION_CHANCE = 0.75;
  static constexpr float PROPAGATION_SIZE_WEIGHT = 1.0;

  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(const Tensor x,
             const char *name = NULL);
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
  // Add a scalar multiply layer
  Tensor scalar_multiply(const Tensor x,
	      const float scalar,
              bool inplace = true,
              const char *name = NULL);
  // Add an activation layer
  Tensor relu(const Tensor x,
              bool inplace = true,
              const char *name = NULL);
  Tensor identity(const Tensor x,
              const char *name = NULL);
  Tensor gelu(const Tensor x,
              const char *name = NULL);
  Tensor sigmoid(const Tensor x,
                 const char *name = NULL);
  Tensor tanh(const Tensor x,
              const char *name = NULL);
  Tensor elu(const Tensor x,
             bool inplace = true,
             const char *name = NULL);
  // Add a 2D convolutional layer
  Tensor conv2d(const Tensor input,
                int outChannels,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                ActiMode activation = AC_MODE_NONE,
                int groups = 1,
                bool use_bias = true,
                const Op* shared_op = NULL,
                Initializer* krenel_initializer = NULL,
                Initializer* bias_initializer = NULL,
                const char* name = NULL);
  // Add a dropout layer
  Tensor dropout(const Tensor input,
                 float rate,
                 unsigned long long seed = 0,
                 const char* name = NULL);
  // Add an embedding layer
  Tensor embedding(const Tensor input,
                   int num_entires, int outDim,
                   AggrMode aggr,
                   const Op* shared_op = NULL,
                   Initializer* kernel_initializer = NULL,
                   const char* name = NULL);
  // Add a group_by layer
  void group_by(const Tensor data,
                const Tensor assign,
                Tensor* outputs,
                int n, float alpha,
                const char* name = NULL);
  // Add aggregate layer
  Tensor aggregate(const Tensor* inputs,
                  int n,
                  const char* name = NULL);
  // Add a 2D pooling layer
  Tensor pool2d(const Tensor input,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE,
                const char* name = NULL);
  // Add a batch_norm layer
  Tensor batch_norm(const Tensor input,
                    bool relu = true,
                    const char* name = NULL);
  // Add a batch_matmul layer
  Tensor batch_matmul(const Tensor A,
                      const Tensor B,
                      int a_seq_length_dim=-1,
                      int b_seq_length_dim=-1);
  // Add a dense layer
  Tensor dense(const Tensor input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               const Op* shared_op = NULL,
               Initializer* kernel_initializer = NULL,
               Initializer* bias_initializer = NULL,
               const char *name = NULL);
  // Add a concat layer
  Tensor concat(int n,
                const Tensor* tensors,
                int axis,
                const char *name = NULL);
  // Add a split layer
  void split(const Tensor input, Tensor* outputs,
             const std::vector<int>& split, int axis,
             const char *name = NULL);
  // Add a flat layer
  Tensor flat(const Tensor input, const char *name = NULL);
  // Add a softmax layer
  Tensor softmax(const Tensor input,
                 int dim=-1,
                 const char *name = NULL);
  // Create input tensors and constants
  Tensor transpose(const Tensor input,
                   const std::vector<int>& perm,
                   const char *name = NULL);
  Tensor reshape(const Tensor input,
                 const std::vector<int>& shape,
                 const char *name = NULL);
  Tensor reverse(const Tensor input,
                 int axis,
                 const char *name = NULL);
  void top_k(const Tensor input,
             Tensor* outputs, int k, bool sorted,
             const char *name = NULL);
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
                             Initializer* kernel_initializer = NULL,
                             const char *name = NULL);
  Tensor create_tensor_legion_ordering(
      int num_dim,
      const int dims[],
      DataType data_type,
      const Op* owner_op = NULL,
      int owner_idx = 0,
      bool create_grad = true);
  Tensor create_tensor_legion_ordering(
      int num_dim,
      const ParallelDim dims[],
      DataType data_type,
      const Op* owner_op = NULL,
      int owner_idx = 0,
      bool create_grad = true);
  Tensor create_tensor(int num_dim,
                       const int dims[],
                       DataType data_type,
                       const Op* owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  Tensor create_tensor(int num_dim,
                       const ParallelDim dims[],
                       DataType data_type,
                       const Op* owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  template<int NDIM>
  Tensor create_tensor(const int dims[],
                       DataType data_type,
                       const Op* owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  template<int NDIM>
  Tensor create_tensor(const ParallelDim dims[],
                       DataType data_type,
                       const Op* owner_op = NULL,
                       int owner_idx = 0,
                       bool create_grad = true);
  template<int NDIM>
  Parameter create_weight(const int dims[],
      DataType data_type,
      const Op* owner_op = NULL,
      bool create_grad = true,
      Initializer* initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  template<int NDIM>
  Parameter create_weight(const ParallelDim dims[],
      DataType data_type,
      const Op* owner_op = NULL,
      bool create_grad = true,
      Initializer* initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  Parameter create_weight(
      int numdim,
      const ParallelDim dims[],
      DataType data_type,
      const Op* owner_op = NULL,
      bool create_grad = true,
      Initializer* initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);
  Parameter create_weight_legion_ordering(
      int numdim,
      const ParallelDim dims[],
      DataType data_type,
      const Op* owner_op = NULL,
      bool create_grad = true,
      Initializer* initializer = NULL,
      ParameterSyncType sync_type = ParameterSyncType::NONE);


  void map_tensor(Tensor tensor, const Op* parallel_op);
  void map_weight(Tensor tensor, const Op* parallel_op);

  template<int NDIM>
  Tensor create_constant(const int dims[],
                         float value,
                         DataType date_type);
  // ========================================
  // Parallel APIs
  // ========================================
  Tensor repartition(
      const Tensor input,
      int partition_legion_dim,
      int partition_degree,
      const char* name = NULL);
  Tensor combine(
      const Tensor input,
      int combine_legion_dim,
      int combine_degree,
      const char* name = NULL);
  Tensor replicate(
      const Tensor input,
      int replicate_legion_dim,
      int replicate_degree,
      const char* name = NULL);
  Tensor reduction(
      const Tensor input,
      int reduction_legion_dim,
      int reduction_degree,
      const char* name = NULL);
  // ========================================
  // Graph APIs
  // ========================================
  float graph_cost(const Graph* graph,
                   const Node& sink_node,
                   const MachineView& sink_view,
                   const Node& source_node,
                   const MachineView& source_view,
                   const MachineResource& resources,
                   bool include_sink_compute_time,
                   bool constructing_optimal_view = false);
  void construct_optimal_view(const Graph* graph,
                              const Node& sink_node,
                              const MachineView& sink_view,
                              const Node& source_node,
                              const MachineView& source_view,
                              const MachineResource& resources,
                              bool include_sink_compute_time,
                              float optimal_cost,
                              std::unordered_map<Node, MachineView>& optimal_views);
  void deserialize_graph_optimal_view(Legion::Deserializer& dez,
                                      Graph* graph,
                                      std::unordered_map<Node, MachineView>& optimal_views);
  bool convert_graph_to_layers(const Graph* graph,
                               const std::unordered_map<Node, MachineView>& optimal_views);
  static void register_all_machine_views(int num_nodes,
                                         int gpus_per_node,
                                         int cpus_per_node,
                                         std::vector<MachineView>& valid_views);
  // ========================================
  // Internal Node creation APIs
  // ========================================
  Node get_or_create_noop_node(const Tensor input);
  Node get_or_create_concat_node(int num_inputs,
                                 const Tensor* inputs,
                                 int axis);
  Node get_or_create_element_binary_node(const Tensor input1,
                                         const Tensor input2,
                                         OperatorType type);
  Node get_or_create_embedding_node(const Tensor input,
                                    int num_entries,
                                    int out_channels,
                                    AggrMode aggr);
  Node get_or_create_linear_node(const Tensor input,
                                 int out_dim,
                                 ActiMode activation,
                                 bool use_bias);
  Node get_or_create_multihead_attn_node(const Tensor query,
                                         const Tensor key,
                                         const Tensor value,
                                         int embed_dim,
                                         int num_heads,
                                         int kdim,
                                         int vdim,
                                         float dropout,
                                         bool bias,
                                         bool add_bias_kv,
                                         bool add_zero_attn);
  Node get_or_create_softmax_node(const Tensor input,
                                  int softmax_dim);
  Node get_or_create_repartition_node(const Tensor input,
                                      int repartition_dim,
                                      int repartition_degree);
  Node get_or_create_replicate_node(const Tensor input,
                                    int replicate_dim,
                                    int replicate_degree);
  Node get_or_create_reduction_node(const Tensor input,
                                    int reduction_dim,
                                    int reduction_degree);
  Node get_or_create_combine_node(const Tensor input,
                                  int combine_dim,
                                  int combine_degree);
  Node get_or_create_fused_parallel_node(const Tensor input,
                                         const std::vector<ParallelOpInfo>& parallel_ops);
  Node get_or_create_conv2d_node(const Tensor input, 
                                 int out_channels,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, 
                                 int padding_h, int padding_w,
                                 ActiMode activation, 
                                 int groups,
                                 bool use_bias);
  Node get_or_create_conv2d_node(const Tensor input,
                                 const Conv2DParams& params);
  Node get_or_create_pool2d_node(const Tensor input,
                                 int kernelH, int kernelW,
                                 int strideH, int strideW,
                                 int paddingH, int paddingW,
                                 PoolType type,
                                 ActiMode activation);
  Node get_or_create_pool2d_node(const Tensor input,
                                 const Pool2DParams& params);
  Node get_or_create_flat_node(const Tensor input);
  Node get_or_create_element_unary_node(const Tensor input,
                                        OperatorType type,
                                        bool inplace, 
                                        float scalar);
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  void create_disjoint_partition(
      int num_dims,
      const ParallelDim dims[],
      const Legion::IndexSpace& part_is,
      const Legion::LogicalRegion& region,
      Legion::LogicalPartition& part);
  template<int NDIM, int TDIM>
  void create_disjoint_partition_with_dim2(
      const ParallelDim dims[],
      const Legion::IndexSpaceT<TDIM>& part_is,
      const Legion::LogicalRegion& region,
      Legion::LogicalPartition& part);
  void create_aliased_partition(
      int num_dims,
      const ParallelDim dims[],
      int aliased_dim,
      const Legion::IndexSpace& part_is,
      const Legion::LogicalRegion& region,
      Legion::LogicalPartition& part);
  template<int NDIM, int TDIM>
  void create_aliased_partition_with_dim2(
      const ParallelDim dims[],
      int aliased_dim,
      const Legion::IndexSpaceT<TDIM>& part_is,
      const Legion::LogicalRegion& region,
      Legion::LogicalPartition& part);

  template<int NDIM>
  void create_disjoint_partition(
      const Tensor tensor,
      const Legion::IndexSpaceT<NDIM>& part_is,
      Legion::LogicalPartition& part_fwd,
      Legion::LogicalPartition& part_bwd);

  template<int NDIM, int TDIM>
  void create_data_parallel_partition_with_diff_dims(
      const Tensor tensor,
      const Legion::IndexSpaceT<TDIM>& task_is,
      Legion::LogicalPartition& part_fwd,
      Legion::LogicalPartition& part_bwd);
  template<int NDIM>
  void map_conv_weight(Tensor p, const Op* parallel_op);
  template<int NDIM, int TDIM>
  void map_linear_weight(Tensor p, const Op* parallel_op);
  template<int NDIM, int TDIM>
  Tensor create_linear_replica(const int* dims,
                               const Legion::IndexSpaceT<TDIM>& part_is,
                               DataType data_type);
  static PerfMetrics update_metrics_task(const Legion::Task *task,
                                         const std::vector<Legion::PhysicalRegion> &regions,
                                         Legion::Context ctx, Legion::Runtime *runtime);
  void reset_metrics();
  void init_layers();
  void prefetch();
  void forward(int seq_length = -1);
  void compute_metrics();
  void backward(int seq_length = -1);
  void update();
  bool apply_fusion(const std::vector<Op*>& layers, std::vector<Op*>& new_layers);
  Op* get_final_layer() const;
  void compile(LossType loss_type,
               const std::vector<MetricsType>& metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void compile(Optimizer* optimizer,
               LossType loss_type,
               const std::vector<MetricsType>& metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void graph_optimize(size_t budget,
                      bool only_data_parallel,
                      Graph*& best_graph,
                      std::unordered_map<Node, MachineView>& optimal_view);
  void mcmc_optimize(std::map<const Op*, ParallelConfig>& best,
                     size_t budget, float alpha,
                     CompMode comp_mode,
                     bool use_propagation) const;
#ifdef FF_USE_NCCL
  ncclComm_t* find_nccl_comms(const MachineView& view) const;
#endif
#ifdef FF_USE_PROPAGATE
  void propagate(std::map<Op *, ParallelConfig> const &current,
                 std::map<Op *, ParallelConfig> &next) const;
#endif
  void rewrite(const std::map<const Op*, ParallelConfig>& current,
               std::map<const Op*, ParallelConfig>& next,
               bool use_propagation) const;
  void zero_gradients();
  void print_layers(int id);
  std::string get_operator_type_name(OperatorType type) const;

  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>> get_bwd_edge_map() const;

  // Internal funcitons
  Legion::IndexSpace get_or_create_task_is(const ParallelConfig& pc);
  Legion::IndexSpace get_or_create_task_is(const MachineView& view);
  Legion::IndexSpace get_or_create_task_is(const Legion::Domain& domain);
  Legion::IndexSpace get_or_create_task_is(const Tensor);
  Legion::IndexSpace get_task_is(const Legion::Domain& domain) const;
  Legion::IndexSpace get_task_is(const ParallelConfig& pc) const;
  Legion::IndexSpace get_task_is(const MachineView& view) const;
  // APIs for setting iteration configs
public:
  void set_iteration_config_sequence_length(int seq_length);
public:
  size_t op_global_guid, tensor_global_guid, node_global_guid;
  FFConfig config;
  FFIterationConfig iter_config;
  Optimizer* optimizer;
  SearchHelper *search;
  Loss* loss_op;
  Metrics* metrics_op;
  Simulator* simulator;
  Tensor label_tensor;

  std::vector<Op*> layers;
  std::vector<Tensor> parameters;
  FFHandler handlers[MAX_NUM_WORKERS];
  Legion::Future current_metrics;
  // Cached operators: key: operator hash, value: operator pointer
  std::unordered_map<size_t, NoOp*> cached_noop_ops;
  std::unordered_map<size_t, Concat*> cached_concat_ops;
  std::unordered_map<size_t, ElementBinary*> cached_element_binary_ops;
  std::unordered_map<size_t, ElementUnary*> cached_element_unary_ops;
  std::unordered_map<size_t, Embedding*> cached_embedding_ops;
  std::unordered_map<size_t, Linear*> cached_linear_ops;
  std::unordered_map<size_t, Conv2D*> cached_conv2d_ops;
  std::unordered_map<size_t, Pool2D*> cached_pool2d_ops;
  std::unordered_map<size_t, Flat*> cached_flat_ops;
  std::unordered_map<size_t, MultiHeadAttention*> cached_multihead_attn_ops;
  std::unordered_map<size_t, Softmax*> cached_softmax_ops;
  std::unordered_map<size_t, Repartition*> cached_repartition_ops;
  std::unordered_map<size_t, Replicate*> cached_replicate_ops;
  std::unordered_map<size_t, Reduction*> cached_reduction_ops;
  std::unordered_map<size_t, Combine*> cached_combine_ops;
  std::unordered_map<size_t, FusedParallelOp*> cached_fused_parallel_ops;
  std::vector<MachineView> all_valid_views;
#ifdef FF_USE_NCCL
  std::unordered_map<size_t, ncclComm_t*> view_hash_to_nccl_comms;
#endif
private:
  bool debug;
  Tensor label_tensor_with_final_part;//FIXME: to be removed
  std::map<MachineView, Legion::IndexSpace, MachineViewDimCompare> all_task_is;

  template<int NDIM>
  void map_tensor_with_dim(Tensor tensor, const Op* parallel_op);
  template<int NDIM, int TDIM>
  void map_tensor_with_dim2(Tensor tensor, const Op* parallel_op);
  template<int NDIM>
  void map_weight_with_dim(Tensor weight, const Op* parallel_op);

  Tensor binary(OperatorType op,
                Tensor const x,
                Tensor const y,
                bool inplace_a = false,
                char const *name = NULL);
  ElementBinary * binary(OperatorType op,
                         char const *name = NULL);
  Tensor unary(OperatorType op,
               Tensor const x,
               bool inplace = true,
               char const *name = NULL,
	       float scalar = 0.0);
  ElementUnary * unary(OperatorType op,
                       char const *name = NULL,
		       float scalar = 0.0);
  Node new_node(Op *);
};

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
  OperatorType op_type;
  bool inplace_a;
};

class NoOp : public Op {
public:
  NoOp(FFModel& model,
       OperatorType type,
       const Tensor output,
       const char* name = NULL);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
};

class ElementBinary : public Op {
public:
  ElementBinary(FFModel& model,
                OperatorType type,
                const Tensor x,
                const Tensor y,
                bool inplace_a,
                const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  bool can_inplace_output();
  bool has_inplace_output();
  void do_inplace_output();

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(const ElementBinaryMeta* m,
                      const float* in1_ptr,
                      const float* in2_ptr,
                      float* out_ptr);
  static void backward_kernel(const ElementBinaryMeta* m,
                       const float* out_grad_ptr,
                       const float* in1_ptr,
                       const float* in2_ptr,
                       float* in1_grad_ptr,
                       float* in2_grad_ptr);
public:
  bool inplace_a;
};

class DropoutMeta;
class Dropout : public Op {
public:
  Dropout(FFModel& model,
          const Tensor input,
          float rate,
          unsigned long long seed,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(DropoutMeta *m,
                             float const *input_ptr,
                             float *output_ptr);
  static void backward_kernel(DropoutMeta *m,
                              float const *output_grad_ptr,
                              float *input_grad_ptr);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  float rate;
  unsigned long long seed;
};

class DropoutMeta : public OpMeta {
public:
  DropoutMeta(FFHandler handle,
              const Dropout* dropout,
              Legion::Memory gpu_mem,
              const Legion::Domain& output_domain);
  ~DropoutMeta(void);
  Realm::RegionInstance reserveInst;
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnDropoutDescriptor_t dropoutDesc;
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
};


class BatchNormMeta;
class BatchNorm : public Op {
public:
  BatchNorm(FFModel& model,
            const Tensor input,
            const Tensor scale,
            const Tensor bias,
            bool relu,
            const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(BatchNormMeta *m,
                             float const *input_ptr,
                             float *output_ptr,
                             float const *scale_ptr,
                             float const *bias_ptr);
  static void backward_kernel(BatchNormMeta *m,
                              float const *input_ptr,
                              float *output_grad_ptr,
                              float const *output_ptr,
                              float *input_grad_ptr,
                              float const *scale_ptr,
                              float *scale_grad_ptr,
                              float *bias_grad_ptr,
                              size_t numElements);
public:
  bool relu;
  int num_replica;
};

class BatchNormMeta : public OpMeta {
public:
  BatchNormMeta(FFHandler handle,
                const BatchNorm* bn,
                Legion::Memory gpu_mem,
                int output_n,
                int output_c,
                int output_h,
                int output_w);
  ~BatchNormMeta(void);
  Realm::RegionInstance reserveInst;
  cudnnTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
};


class BatchMatmulMeta : public OpMeta {
public:
  BatchMatmulMeta(FFHandler handler);
  int a_seq_length_dim, b_seq_length_dim;
};

class BatchMatmul : public Op {
public:
  BatchMatmul(FFModel& model,
              const Tensor A,
              const Tensor B,
              int a_seq_length_dim,
              int b_seq_length_dim);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const BatchMatmulMeta* meta,
                      float* o_ptr,
                      const float* a_ptr,
                      const float* b_ptr,
                      const float* c_ptr,
                      int m, int n, int k,
                      int batch,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      int seq_length = -1);
  static void backward_kernel(const BatchMatmulMeta* meta,
                       const float* o_ptr,
                       const float* o_grad_ptr,
                       const float* a_ptr,
                       float* a_grad_ptr,
                       const float* b_ptr,
                       float* b_grad_ptr,
                       float* c_grad_ptr,
                       int m, int n, int k, int batch);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
private:
  template<int NDIM>
  void init_with_dim(const FFModel& ff);
  template<int NDIM>
  void forward_with_dim(const FFModel& ff);
  template<int NDIM>
  void backward_with_dim(const FFModel& ff);
public:
  int a_seq_length_dim, b_seq_length_dim;
};

class Group_byMeta : public OpMeta {
public:
  Group_byMeta(FFHandler handle);
};

class Group_by : public Op {
public:
  Group_by(FFModel& model,
          const Tensor _input,
          const Tensor _assign,
          int _n, float _alpha,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int n;
  float alpha;
};


class AggregateMeta : public OpMeta {
public:
  AggregateMeta(FFHandler handle);
};

class Aggregate : public Op {
public:
  Aggregate(FFModel& model,
            const Tensor* inputs,
            int _n, const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int n;
};

class SoftmaxMeta;
class Softmax : public Op {
public:
  Softmax(FFModel& model,
          const Tensor logit,
          int dim,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  bool get_int_parameter(PMParameter, int*) const;
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  void init_meta(SoftmaxMeta *m,
                 Legion::Rect<2> const &input,
                 Legion::Rect<2> const &output) const;
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(SoftmaxMeta const *m,
                             float const *input_ptr,
                             float *output_ptr);
  static void backward_kernel(float *input_grad_ptr,
                              float const *output_grad_ptr,
                              size_t num_elements);
private:
  template<int NDIM>
  static void forward_task_with_dim(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx, Legion::Runtime *runtime);
  template<int NDIM>
  static void backward_task_with_dim(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &regions,
                                     Legion::Context ctx, Legion::Runtime *runtime);
public:
  int dim;
};

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle,
              const Softmax* softmax,
              const Legion::Domain& input_domain);
  cudnnTensorDescriptor_t inputTensor;
  bool profiling;
  int dim;
  char op_name[MAX_OPNAME];
};

class TransposeMeta : public OpMeta {
public:
  TransposeMeta(FFHandler handler) : OpMeta(handler) {};
  int num_dim;
  int perm[MAX_TENSOR_DIM];
};

class Transpose : public Op {
public:
  Transpose(FFModel& model,
            const Tensor input,
            const std::vector<int>& perm,
            const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  void init_meta(TransposeMeta *m,
                 Legion::Domain const &in_domain,
                 Legion::Domain const &out_domain) const;
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const TransposeMeta* m,
                             const float* input_ptr,
                             float* output_ptr,
                             Legion::Domain in_domain,
                             Legion::Domain out_domain);
  static void backward_kernel(const TransposeMeta* m,
                              float* input_grad_ptr,
                              const float* output_grad_ptr,
                              Legion::Domain in_grad_domain,
                              Legion::Domain out_grad_domain);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int perm[MAX_TENSOR_DIM];
};

class Reverse : public Op {
public:
  Reverse(FFModel& model,
          const Tensor input,
          int axis,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             Legion::coord_t num_out_blks,
                             Legion::coord_t reverse_dim_size,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t output_size);
  static void backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              Legion::coord_t num_out_blks,
                              Legion::coord_t reverse_dim_size,
                              Legion::coord_t in_blk_size,
                              Legion::coord_t input_size);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int axis;
};

class Reshape : public Op {
public:
  Reshape(FFModel& model,
          const Tensor input,
          const std::vector<int>& shape,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements);
  static void backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
};

class TopKMeta : public OpMeta {
public:
  TopKMeta(FFHandler handle);
  bool sorted;
};

class TopK : public Op {
public:
  TopK(FFModel& model,
       const Tensor input,
       int k, bool sorted,
       const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(const TopKMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      int* indices_ptr,
                      size_t batch_size, int length, int k,
                      bool sorted);
  static void backward_kernel(const TopKMeta* m,
                       const float* out_grad_ptr,
                       const int* indices_ptr,
                       float* in_grad_ptr,
                       size_t batch_size, int length, int k);
public:
  int k;
  bool sorted;
};

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
  int axis;
};

class Concat : public Op {
public:
  Concat(FFModel& model,
         int n,
         const Tensor* inputs,
         int axis,
         const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  bool get_int_parameter(PMParameter, int*) const;
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  void init_meta(ConcatMeta *meta) const;
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(float* output,
                             float const * const *inputs,
                             int num_inputs,
                             int axis,
                             const Legion::Domain& out_domain,
                             const Legion::Domain* in_domain);
  static void backward_kernel(const float* output_grad,
                              float** input_grads,
                              int num_inputs,
                              int axis,
                              const Legion::Domain& out_grad_domain,
                              const Legion::Domain* in_grad_domain);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int axis;
};

class Split : public Op {
public:
  Split(FFModel& model,
        const Tensor input,
        const std::vector<int>& split,
        int axis,
        const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(float **out_ptrs,
                             float const *in_ptr,
                             Legion::coord_t const *out_blk_sizes,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t num_blks,
                             int numOutputs);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int axis;
};

class FusedOp;
class FusedOpMeta {
public:
  FusedOpMeta(void) {}
  OpMeta* meta[MAX_NUM_FUSED_OPERATORS];
  FusedOp* fused_op;
  int numOperators;
};

class FusedOp : public Op {
public:
  enum SourceType {
    SOURCE_NONE,
    SOURCE_INPUT,
    SOURCE_WEIGHT,
    SOURCE_OUTPUT,
  };
  FusedOp(FFModel& model,
          Op* op);
  bool add_operator(FFModel& model, Op* op);
  Tensor init_inout(FFModel& model, const Tensor input) {assert(0); return Tensor();}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  FFIterationConfig iter_config;
  int op_num_inputs[MAX_NUM_FUSED_OPERATORS];
  int op_num_weights[MAX_NUM_FUSED_OPERATORS];
  int op_num_outputs[MAX_NUM_FUSED_OPERATORS];
  OperatorType op_op_type[MAX_NUM_FUSED_OPERATORS];
  SourceType op_input_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_weight_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_output_source[MAX_NUM_FUSED_TENSORS];
  int op_input_idx[MAX_NUM_FUSED_TENSORS];
  int op_weight_idx[MAX_NUM_FUSED_TENSORS];
  int op_output_idx[MAX_NUM_FUSED_TENSORS];
  Op* operators[MAX_NUM_FUSED_OPERATORS];
  FusedOpMeta fused_meta[MAX_NUM_WORKERS];
  int numOperators;
};

class ParallelOp : public Op {
public:
  ParallelOp(FFModel& model,
             OperatorType type,
             const char* _name,
             const Tensor input);
  virtual void init(const FFModel&) = 0;
  virtual void forward(const FFModel&) = 0;
  virtual void backward(const FFModel&) = 0;
  virtual void create_input_partition(FFModel& model) = 0;
  void print_layer(const FFModel& model) {};
  virtual bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const = 0;
  virtual bool append_parallel_op_info(std::vector<ParallelOpInfo>& parallel_ops) const = 0;
  virtual bool is_parallel_op() const;
public:
  Legion::LogicalPartition input_lp, output_grad_lp;
};


class UtilityTasks {
public:
  static FFHandler init_cuda_task(const Legion::Task *task,
                                  const std::vector<Legion::PhysicalRegion> &regions,
                                  Legion::Context ctx, Legion::Runtime *runtime);
  static void dummy_task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx, Legion::Runtime *runtime);
  static void init_images_task(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx, Legion::Runtime *runtime);
  static void init_labels_task(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx, Legion::Runtime *runtime);
  static void load_images_task(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx, Legion::Runtime *runtime);
  static void normalize_images_task(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx, Legion::Runtime *runtime);
};

void top_level_task(const Legion::Task* task,
                    const std::vector<Legion::PhysicalRegion>& regions,
                    Legion::Context ctx, Legion::Runtime* runtime);

void data_load_task(const Legion::Task* task,
                    const std::vector<Legion::PhysicalRegion>& regions,
                    Legion::Context ctx, Legion::Runtime* runtime);

void register_flexflow_internal_tasks();

void register_custom_tasks();
#endif//_FLEXFLOW_MODEL_H_

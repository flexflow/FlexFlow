#ifndef _OPERATOR_H
#define _OPERATOR_H

#include "flexflow/accessor.h"
#include "flexflow/batch_config.h"
#include "flexflow/fftype.h"
#include "flexflow/machine_view.h"
#include "flexflow/parallel_tensor.h"
#include "flexflow/utils/dot/record_formatter.h"
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_measure;

class OpMeta;
class Simulator;
class CostMetrics;

enum class MappingRecordType { INPUT_OUTPUT, INPUT_WEIGHT };

enum class MappingOperation { PARTITION, REPLICATE };

/** @brief  A class to keep track of a dimension relation between two tensors
 * used by an operator.
 *
 * Dimension relations are one-to-one mappings between the dimensions of the
 * input, weights, and output tensors of an operator. Introduced in the Unity
 * paper, dimension relations allow FlexFlow to keep track of an operator's
 * parallelization plans as part of the Parallel Computation Graph (PCG).
 *
 * Each ParallelDimMappingRecord only keeps track of a single dimension
 * relation.
 *
 * ParallelDimMappingRecord objects must be initialized with a
 * MappingRecordType, which can be INPUT_OUTPUT, if the ParallelDimMappingRecord
 * is tracking a dimension relation between the input and the output tensor, or
 * INPUT_WEIGHT, if the ParallelDimMappingRecord is tracking a dimension
 * relation between the input tensor and the weights tensor.
 *
 */
class ParallelDimMappingRecord {
private:
  ParallelDimMappingRecord(MappingRecordType);

public:
  /**
   * @brief We disable this constructor because ParallelDimMappingRecord objects
   * must specify the MappingRecordType upon creation.
   */
  ParallelDimMappingRecord() = delete;

  static ParallelDimMappingRecord input_output_record(
      int input_idx,
      int input_dim,
      int output_idx,
      int output_dim,
      tl::optional<MappingOperation> operation = tl::nullopt);
  static ParallelDimMappingRecord input_weight_record(
      int input_idx,
      int input_dim,
      int weight_idx,
      int weight_dim,
      tl::optional<MappingOperation> operation = tl::nullopt);
  MappingRecordType get_type() const;

public:
  MappingRecordType type;
  tl::optional<MappingOperation> operation;

  int output_dim, input_dim, weight_dim;
  int output_idx, input_idx, weight_idx;
};

class Op {
public:
  static void construct_weight_parallel_dims(
      std::vector<ParallelDimMappingRecord> &records,
      std::vector<std::pair<int, int>> mappings,
      int input_idx = 0,
      int weight_idx = 0);
  static void construct_weight_parallel_dims(
      std::vector<ParallelDimMappingRecord> &records,
      std::vector<std::tuple<int, MappingOperation, int>> mappings,
      int input_idx = 0,
      int weight_idx = 0);
  static void construct_weight_parallel_dims(
      std::vector<ParallelDimMappingRecord> &records,
      int input_dim,
      int weight_dim,
      int input_idx = 0,
      int weight_idx = 0,
      tl::optional<MappingOperation> operation = tl::nullopt);

  static void construct_output_parallel_dims(
      std::vector<ParallelDimMappingRecord> &records,
      std::vector<std::pair<int, int>> mappings,
      int input_idx = 0,
      int output_idx = 0);
  static void construct_output_parallel_dims(
      std::vector<ParallelDimMappingRecord> &records,
      std::vector<std::tuple<int, MappingOperation, int>> mappings,
      int input_idx = 0,
      int output_idx = 0);
  static void construct_output_parallel_dims(
      std::vector<ParallelDimMappingRecord> &records,
      int input_dim,
      int output_dim,
      int input_idx = 0,
      int output_idx = 0,
      tl::optional<MappingOperation> operation = tl::nullopt);

  ParallelConfig view_to_pc(MachineView const &view) const;

protected:
  void register_weight_parallel_dims(std::vector<std::pair<int, int>> mappings,
                                     int input_idx = 0,
                                     int weight_idx = 0);
  void register_weight_parallel_dims(
      std::vector<std::tuple<int, MappingOperation, int>> mappings,
      int input_idx = 0,
      int weight_idx = 0);
  void register_weight_parallel_dims(
      int input_dim,
      int weight_dim,
      int input_idx = 0,
      int weight_idx = 0,
      tl::optional<MappingOperation> operation = tl::nullopt);

  void register_output_parallel_dims(std::vector<std::pair<int, int>> mappings,
                                     int input_idx = 0,
                                     int output_idx = 0);
  void register_output_parallel_dims(
      std::vector<std::tuple<int, MappingOperation, int>> mappings,
      int input_idx = 0,
      int output_idx = 0);
  void register_output_parallel_dims(
      int input_dim,
      int output_dim,
      int input_idx = 0,
      int output_idx = 0,
      tl::optional<MappingOperation> operation = tl::nullopt);

  int get_output_to_input_dim_mapping(const ParallelTensor output,
                                      int output_dim,
                                      const ParallelTensor input);
  int get_output_to_weight_dim_mapping(const ParallelTensor output,
                                       int output_dim,
                                       const ParallelTensor weight);

  void inner_measure_operator_cost(Simulator *sim,
                                   std::function<void()> const &forward,
                                   std::function<void()> const &backward,
                                   CostMetrics &cost_metrics) const;

  bool check_output_input_weight_parallel_dims(
      bool allocate_weights = true) const;
  bool check_output_input_weight_same_parallel_is() const;
  bool check_output_input_weight_same_machine_view() const;

public:
  Op(FFModel &model,
     OperatorType otype,
     DataType dtype,
     char const *_name,
     int numInputs,
     int numWeights,
     bool allocate_weights,
     int numOutputs,
     const ParallelTensor input1 = NULL,
     const ParallelTensor input2 = NULL,
     const ParallelTensor input3 = NULL,
     const ParallelTensor input4 = NULL);
  Op(FFModel &model,
     OperatorType otype,
     DataType dtype,
     char const *_name,
     int numInputs,
     int numWeights,
     int numOutputs,
     const ParallelTensor input1 = NULL,
     const ParallelTensor input2 = NULL,
     const ParallelTensor input3 = NULL,
     const ParallelTensor input4 = NULL);
  Op(int guid,
     bool profiling,
     bool inference_debugging,
     OperatorType otype,
     DataType dtype,
     char const *name,
     int numInputs,
     int numWeights,
     int numOutputs,
     const ParallelTensor input1 = NULL,
     const ParallelTensor input2 = NULL,
     const ParallelTensor input3 = NULL,
     const ParallelTensor input4 = NULL);
  Op(FFModel &model,
     OperatorType otype,
     DataType dtype,
     char const *_name,
     int numInputs,
     int numWeights,
     int numOutputs,
     ParallelTensor const *tensors);
  // graph substitution related methods
  virtual bool get_int_parameter(PMParameter, int *) const;
  virtual bool get_tensor_parameter(TNParameter, DIMParameter, int *) const;
  virtual bool get_input_parameter(TNParameter, DIMParameter, int *) const;
  virtual bool get_weight_parameter(TNParameter, DIMParameter, int *) const;
  // Pure virtual functions that must be implemented
  virtual void init(FFModel const &) = 0;
  virtual void init_inference(FFModel const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) {
    assert(false);
  };
  virtual void forward(FFModel const &) = 0;
  virtual void backward(FFModel const &) = 0;
  // Pure virtual functions for inference
  virtual Legion::FutureMap inference(FFModel const &,
                                      BatchConfigFuture const &,
                                      std::vector<ParallelTensor> const &,
                                      std::vector<ParallelTensor> const &,
                                      MachineView const *mv = nullptr) {
    assert(false);
  };
  virtual void print_layer(FFModel const &model) = 0;
  template <typename OpMetaType>
  static std::string get_op_name_without_uid(OpMetaType *m) {
    std::string op_name_without_uid = std::string(m->op_name);
    size_t last_underscore = op_name_without_uid.length() - 1;
    for (int i = op_name_without_uid.length() - 1; i > 0; i--) {
      if (!(std::isdigit(m->op_name[i]) || m->op_name[i] == '_')) {
        break;
      } else if (m->op_name[i] == '_') {
        last_underscore = i;
      }
    }
    op_name_without_uid.erase(last_underscore);
    return op_name_without_uid;
  }
  template <typename OpMetaType>
  static void save_inference_tensors_to_file(
      OpMetaType *m,
      int shard_id,
      BatchConfig const *bc,
      std::vector<GenericTensorAccessorR> input_tensors,
      std::vector<GenericTensorAccessorR> weight_tensors,
      std::vector<GenericTensorAccessorR> output_tensors,
      bool before_kernel = false) {
    // Check if output directory exists, and create it if it does not
    char const *folder_path = "./inference_tensors";
    struct stat st = {0};
    if (stat(folder_path, &st) == -1) {
      // Directory does not exist, create it
      mkdir(folder_path, 0700);
    }
    // output base filepath, shared by all tensors from the same operator
    std::string op_name_without_uid = get_op_name_without_uid(m);
    std::string base_filepath =
        "./inference_tensors/model_" + std::to_string(m->layer_guid.model_id) +
        "_decoding-step_" + std::to_string(m->decoding_step) + "_layer-num_" +
        std::to_string(m->layer_guid.transformer_layer_id) + "_layer-name_" +
        op_name_without_uid + "_shard-id_" + std::to_string(shard_id);
    if (before_kernel) {
      base_filepath += "_pre";
    }
    // save batch config, if passed
    if (bc != nullptr) {
      bc->save_to_file(base_filepath + "_batch-config");
    }
    // save all inputs
    for (int i = 0; i < input_tensors.size(); i++) {
      std::string filename = base_filepath + "_input_" + std::to_string(i);
      if (input_tensors[i].data_type == DT_FLOAT) {
        save_tensor(input_tensors[i].get_float_ptr(),
                    input_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (input_tensors[i].data_type == DT_HALF) {
        save_tensor(input_tensors[i].get_half_ptr(),
                    input_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (input_tensors[i].data_type == DT_INT32) {
        save_tensor(input_tensors[i].get_int32_ptr(),
                    input_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (input_tensors[i].data_type == DT_INT64) {
        save_tensor(input_tensors[i].get_int64_ptr(),
                    input_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else {
        assert(false && "Tensor data type not supported");
      }
    }
    // only dump the weights once
    if (m->decoding_step == 0) {
      for (int i = 0; i < weight_tensors.size(); i++) {
        std::string filename = base_filepath + "_weight_" + std::to_string(i);
        if (weight_tensors[i].data_type == DT_FLOAT) {
          save_tensor(weight_tensors[i].get_float_ptr(),
                      weight_tensors[i].domain.get_volume(),
                      filename.c_str());
        } else if (weight_tensors[i].data_type == DT_HALF) {
          save_tensor(weight_tensors[i].get_half_ptr(),
                      weight_tensors[i].domain.get_volume(),
                      filename.c_str());
        } else if (weight_tensors[i].data_type == DT_INT32) {
          save_tensor(weight_tensors[i].get_int32_ptr(),
                      weight_tensors[i].domain.get_volume(),
                      filename.c_str());
        } else if (weight_tensors[i].data_type == DT_INT64) {
          save_tensor(weight_tensors[i].get_int64_ptr(),
                      weight_tensors[i].domain.get_volume(),
                      filename.c_str());
        } else {
          assert(false && "Tensor data type not supported");
        }
      }
    }
    // save all outputs
    for (int i = 0; i < output_tensors.size(); i++) {
      std::string filename = base_filepath + "_output_" + std::to_string(i);
      if (output_tensors[i].data_type == DT_FLOAT) {
        save_tensor(output_tensors[i].get_float_ptr(),
                    output_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (output_tensors[i].data_type == DT_HALF) {
        save_tensor(output_tensors[i].get_half_ptr(),
                    output_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (output_tensors[i].data_type == DT_INT32) {
        save_tensor(output_tensors[i].get_int32_ptr(),
                    output_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (output_tensors[i].data_type == DT_INT64) {
        save_tensor(output_tensors[i].get_int64_ptr(),
                    output_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else {
        assert(false && "Tensor data type not supported");
      }
    }
    // increase count of decoding steps
    if (!before_kernel) {
      m->decoding_step++;
    }
  }
  virtual bool measure_operator_cost(Simulator *sim,
                                     MachineView const &mv,
                                     CostMetrics &cost_metrics) const = 0;
  virtual bool estimate_sync_cost(Simulator *sim,
                                  MachineView const &pc,
                                  CostMetrics &cost_metrics) const;
  // Other virtual functions that can be optionally overwritten
  virtual ParallelConfig get_random_parallel_config(FFModel const &ff) const;
  virtual ParallelConfig get_data_parallel_config(FFModel const &ff) const;
  virtual Legion::Domain get_input_tensor_shape(ParallelConfig const &pc,
                                                int input_idx,
                                                int part_idx) const;
  virtual Legion::Domain get_output_tensor_shape(ParallelConfig const &pc,
                                                 int output_idx,
                                                 int part_idx) const;
  virtual Legion::Domain get_weight_tensor_shape(ParallelConfig const &pc,
                                                 int weight_idx,
                                                 int part_idx) const;
  virtual bool is_valid_parallel_config(FFModel const &ff,
                                        ParallelConfig const &pc) const;
  virtual bool is_adoptable_parallel_config(FFModel const &ff,
                                            ParallelConfig const &pc) const;
  // Helper functions
  void prefetch(FFModel const &);
  void zero_grad(FFModel const &);
  ParallelTensor get_parameter(int index);
  virtual void map_output_tensors(FFModel &ff);
  virtual bool can_inplace_output();
  virtual bool has_inplace_output();
  virtual void do_inplace_output();
  virtual bool is_parallel_op() const;
  virtual void serialize(Legion::Serializer &) const;
  virtual Op *
      materialize(FFModel &ff, ParallelTensor inputs[], int num_inputs) const;
  size_t get_untyped_params_hash() const;
  virtual size_t get_params_hash() const;

  virtual tl::optional<RecordFormatter> as_dot() const;

  int get_dimension() const;
#ifdef FF_USE_NCCL
  static ncclUniqueId get_nccl_unique_id_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  static ncclComm_t
      init_nccl_comms_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void
      finish_nccl_comms_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
#endif
protected:
  void set_argumentmap_for_init(FFModel const &ff, Legion::ArgumentMap &argmap);
  void set_argumentmap_for_init_inference(FFModel const &ff,
                                          Legion::ArgumentMap &argmap,
                                          ParallelTensor const output0);
  void set_argumentmap_for_forward(FFModel const &ff,
                                   Legion::ArgumentMap &argmap);
  void set_argumentmap_for_inference(FFModel const &ff,
                                     Legion::ArgumentMap &argmap,
                                     ParallelTensor const output0);
  void set_argumentmap_for_backward(FFModel const &ff,
                                    Legion::ArgumentMap &argmap);
  void set_opmeta_from_futuremap(FFModel const &ff,
                                 Legion::FutureMap const &fm);
  void set_opmeta_from_futuremap_inference(FFModel const &ff,
                                           Legion::FutureMap const &fm,
                                           ParallelTensor const output0);
  void solve_parallel_dim_mappings(
      std::vector<ParallelDim const *> const &inputs,
      std::vector<ParallelDim *> const &weights,
      std::vector<ParallelDim *> const &outputs) const;

public:
  OperatorType op_type;
  DataType data_type;
  // the guid of the layer associated with the current operator
  // layer_guid is used to match layer with op
  LayerID layer_guid;
  size_t op_guid;
  char name[MAX_OPNAME];
  Legion::IndexSpace parallel_is;
  ParallelTensor outputs[MAX_NUM_OUTPUTS];
  ParallelTensor inputs[MAX_NUM_INPUTS];
  ParallelParameter weights[MAX_NUM_WEIGHTS];
  bool trainableInputs[MAX_NUM_INPUTS];
  OpMeta *meta[MAX_NUM_WORKERS];
  std::map<ParallelTensor, OpMeta *[MAX_NUM_WORKERS]> inference_meta;
  int numInputs, numWeights, numOutputs;
  bool profiling;
  bool inference_debugging;
  bool add_bias_only_once;
#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
#endif
  // Note: parallel_dims_mapping should not be called in a DNN task
  std::vector<ParallelDimMappingRecord> *parallel_dims_mapping;
};

}; // namespace FlexFlow

#endif // _OPERATOR_H

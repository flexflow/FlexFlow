#ifndef _OPERATOR_H
#define _OPERATOR_H

#include "flexflow/fftype.h"
#include "flexflow/machine_view.h"
#include "flexflow/parallel_tensor.h"
#include "flexflow/utils/dot/record_formatter.h"
#include <vector>

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_measure;

class OpMeta;
class Simulator;
class CostMetrics;

enum class MappingRecordType { INPUT_OUTPUT, INPUT_WEIGHT };

enum class MappingOperation { PARTITION, REPLICATE };

class ParallelDimMappingRecord {
private:
  ParallelDimMappingRecord(MappingRecordType);

public:
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
     OperatorType type,
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
     OperatorType type,
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
     OperatorType type,
     char const *name,
     int numInputs,
     int numWeights,
     int numOutputs,
     const ParallelTensor input1 = NULL,
     const ParallelTensor input2 = NULL,
     const ParallelTensor input3 = NULL,
     const ParallelTensor input4 = NULL);
  Op(FFModel &model,
     OperatorType type,
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
  virtual void forward(FFModel const &) = 0;
  virtual void backward(FFModel const &) = 0;
  virtual void print_layer(FFModel const &model) = 0;
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
#endif
protected:
  void set_argumentmap_for_init(FFModel const &ff, Legion::ArgumentMap &argmap);
  void set_argumentmap_for_forward(FFModel const &ff,
                                   Legion::ArgumentMap &argmap);
  void set_argumentmap_for_backward(FFModel const &ff,
                                    Legion::ArgumentMap &argmap);
  void set_opmeta_from_futuremap(FFModel const &ff,
                                 Legion::FutureMap const &fm);
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
  int numInputs, numWeights, numOutputs;
  bool profiling;
#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
#endif
  // Note: parallel_dims_mapping should not be called in a DNN task
  std::vector<ParallelDimMappingRecord> *parallel_dims_mapping;
};

}; // namespace FlexFlow

#endif // _OPERATOR_H

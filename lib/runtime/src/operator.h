#ifndef _OPERATOR_H
#define _OPERATOR_H

#include "runtime/config.h"
#include "fftype.h"
#include "pcg/machine_view.h"
#include "parallel_tensor.h"
#include <vector>
#include "utils/stack_vector.h"
#include "model.h"
#include "runtime/tasks.h"

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_measure;

class OpMeta;
class Simulator;
class CostMetrics;

enum class TensorRole {
  INPUT,
  PARAM,
  OUTPUT,
};

enum class IsGrad {
  YES,
  NO
};

enum class Pass {
  FWD,
  BWD
};

struct TensorSpec {
  TensorSpec() = delete;
  TensorSpec(TensorRole, int, IsGrad is_grad = IsGrad::NO, tl::optional<Legion::PrivilegeMode> mode = tl::nullopt);

  TensorRole role;
  int idx;
  IsGrad is_grad;
  tl::optional<Legion::PrivilegeMode> mode;
};

Legion::PrivilegeMode get_default_mode(Pass, TensorRole, IsGrad);

struct TaskSpec {
  template <typename T>
  TaskSpec(TaskID task_id, Pass pass, std::vector<TensorSpec> const &tensors, T const &arg)
    : TaskSpec(task_id, pass, tensors, Legion::TaskArgument{&arg.value(), sizeof(T)})
  { }

  template <>
  TaskSpec(TaskID task_id, Pass pass, std::vector<TensorSpec> const &tensors, Legion::TaskArgument const &argument)
    : task_id(task_id), pass(pass), argument(argument), tensors(tensors)
  { 
    for (TensorSpec &tensor_spec : this->tensors) {
      tensor_spec.mode = tensor_spec.mode.value_or(get_default_mode(pass, tensor_spec.role, tensor_spec.is_grad));
    }
  }

  TaskSpec(TaskID task_id, Pass pass, std::vector<TensorSpec> const &tensors) 
    : TaskSpec(task_id, pass, tensors, Legion::TaskArgument{nullptr, 0})
  { }

  template <typename ...Ts>
  void add_tensor(Ts const &...ts) {
    this->tensors.push_back(TensorSpec{ts...});
  }

  TaskID task_id;
  Pass pass;
  Legion::TaskArgument argument;
  std::vector<TensorSpec> tensors;
};

class Op {
protected:
  void register_weight_parallel_dims(std::vector<std::pair<int, int>> mappings,
                                     int input_idx = 0,
                                     int weight_idx = 0);

  void register_output_parallel_dims(std::vector<std::pair<int, int>> mappings,
                                     int input_idx = 0,
                                     int output_idx = 0);

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
     tl::optional<ParallelTensor> const &input1 = tl::nullopt,
     tl::optional<ParallelTensor> const &input2 = tl::nullopt,
     tl::optional<ParallelTensor> const &input3 = tl::nullopt,
     tl::optional<ParallelTensor> const &input4 = tl::nullopt);
  Op(FFModel &model,
     OperatorType otype,
     DataType dtype,
     char const *_name,
     int numInputs,
     int numWeights,
     int numOutputs,
     tl::optional<ParallelTensor> const &input1 = tl::nullopt,
     tl::optional<ParallelTensor> const &input2 = tl::nullopt,
     tl::optional<ParallelTensor> const &input3 = tl::nullopt,
     tl::optional<ParallelTensor> const &input4 = tl::nullopt);
  Op(int guid,
     bool profiling,
     OperatorType otype,
     DataType dtype,
     char const *name,
     int numInputs,
     int numWeights,
     int numOutputs,
     tl::optional<ParallelTensor> const &input1 = tl::nullopt,
     tl::optional<ParallelTensor> const &input2 = tl::nullopt,
     tl::optional<ParallelTensor> const &input3 = tl::nullopt,
     tl::optional<ParallelTensor> const &input4 = tl::nullopt);
  Op(FFModel &model,
     OperatorType otype,
     DataType dtype,
     char const *_name,
     int numInputs,
     int numWeights,
     int numOutputs,
     ParallelTensor const *tensors);
  
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
  virtual MachineView get_random_parallel_config(FFModel const &ff) const;
  virtual MachineView get_data_parallel_config(FFModel const &ff) const;
  virtual Legion::Domain get_input_tensor_shape(MachineView const &pc,
                                                int input_idx,
                                                int part_idx) const;
  virtual Legion::Domain get_output_tensor_shape(MachineView const &pc,
                                                 int output_idx,
                                                 int part_idx) const;
  virtual Legion::Domain get_weight_tensor_shape(MachineView const &pc,
                                                 int weight_idx,
                                                 int part_idx) const;
  virtual bool is_valid_parallel_config(FFModel const &ff,
                                        MachineView const &pc) const;
  virtual bool is_adoptable_parallel_config(FFModel const &ff,
                                            MachineView const &pc) const;
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
                                   Legion::ArgumentMap &argmap) const;
  void set_argumentmap_for_backward(FFModel const &ff,
                                    Legion::ArgumentMap &argmap);
  void set_opmeta_from_futuremap(FFModel const &ff,
                                 Legion::FutureMap const &fm);

  template <typename T>
  Legion::IndexLauncher make_fwd_index_launcher(FFModel const &ff, TaskID task_id, tl::optional<T const &> arg = tl::nullopt) const {
    using namespace Legion;

  }

  void require_input_tensor(Legion::IndexLauncher &, int idx) const;
  void require_output_tensor(Legion::IndexLauncher &, int idx) const;
  void require_weight_tensor(Legion::IndexLauncher &, int idx) const;

  virtual TaskSpec get_forward_task_spec() const = 0;
  virtual TaskSpec get_backward_task_spec() const = 0;

  void execute_task_spec(FFModel const &, TaskSpec const &);
  ParallelTensor const &get_parallel_tensor(TensorRole, int); 
public:
  OperatorType op_type;
  DataType data_type;
  // the guid of the layer associated with the current operator
  // layer_guid is used to match layer with op
  LayerID layer_guid;
  size_t op_guid;
  char name[MAX_OPNAME];
  Legion::IndexSpace parallel_is;
  stack_vector<ParallelTensor, MAX_NUM_OUTPUTS> outputs;
  stack_vector<ParallelTensor, MAX_NUM_INPUTS> inputs;
  stack_vector<ParallelParameter, MAX_NUM_WEIGHTS> weights;
  stack_vector<bool, MAX_NUM_INPUTS> trainableInputs;
  stack_vector<OpMeta *, MAX_NUM_WORKERS> meta;
  int numInputs, numWeights, numOutputs;
  bool profiling;
#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
#endif
};

}

#endif 

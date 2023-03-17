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
#include <stdexcept>
#include "task_spec.h"

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_measure;

class OpMeta;
class Simulator;
class CostMetrics;

/* struct TaskSpec { */
/* private: */
/*   struct AddTensorResult { */
/*     AddTensorResult() = delete; */
/*     explicit AddTensorResult(int idx); */

/*     int idx; */
/*   }; */

/* public: */
/*   template <typename T> */
/*   TaskSpec(TaskID task_id, OpTaskType task_type, std::vector<TensorSpec> const &tensors, T const &arg) */
/*     : TaskSpec(task_id, task_type, tensors, Legion::TaskArgument{&arg.value(), sizeof(T)}) */
/*   { } */

/*   template <> */
/*   TaskSpec(TaskID task_id, OpTaskType task_type, std::vector<TensorSpec> const &tensors, Legion::TaskArgument const &argument) */
/*     : task_id(task_id), task_type(task_type), argument(argument), tensors(tensors) */
/*   { */ 
/*     for (TensorSpec &tensor_spec : this->tensors) { */
/*       tensor_spec.mode = tensor_spec.mode.value_or(get_default_mode(task_type, tensor_spec.role, tensor_spec.is_grad)); */
/*     } */
/*   } */

/*   TaskSpec(TaskID task_id, Pass pass, std::vector<TensorSpec> const &tensors) */ 
/*     : TaskSpec(task_id, pass, tensors, Legion::TaskArgument{nullptr, 0}) */
/*   { } */

/*   template <typename ...Ts> */
/*   AddTensorResult add_tensor(int name, Ts const &...ts) { */
/*     this->tensors.push_back(TensorSpec{ts...}); */
/*     return AddTensorResult(this->tensors.size()-1); */
/*   } */

/*   AddTensorResult map_tensor_to_name(AddTensorResult const &, int name); */
/*   AddTensorResult map_tensor_to_names(AddTensorResult const &, std::unordered_set<int> const &names); */

/*   template <typename ...Ts> */
/*   AddTensorResult add_named_tensor(int name, Ts const &...ts) { */
/*     return this->add_tensor(std::unordered_set<int>{name}, ts...); */
/*   } */

/*   template <typename ...Ts> */
/*   AddTensorResult add_named_tensor(std::unordered_set<int> const &names, Ts const &...ts) { */
/*     auto result = this->add_tensor(ts...); */
/*     this->map_tensor_to_names(result, names); */
/*   } */

/*   std::pair<int, TensorSpec> get_tensor_spec_by_name(int name) const; */

/*   AddTensorResult &operator[](int name); */

/*   TaskID task_id; */
/*   Pass pass; */
/*   Legion::TaskArgument argument; */

/* private: */
/*   std::vector<TensorSpec> tensors; */
/*   std::unordered_map<int, int> name_map; */
/* }; */

class Op {
protected:
  void inner_measure_operator_cost(Simulator *sim,
                                   std::function<void()> const &forward,
                                   std::function<void()> const &backward,
                                   CostMetrics &cost_metrics) const;

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
  virtual void init(FFModel const &);
  virtual void forward(FFModel const &);
  virtual void backward(FFModel const &);

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

  virtual OpTasksSpec get_tasks_spec() const = 0;
  OpTasksSpec get_fully_defined_tasks_spec() const;
  OpTaskSpec infer_bwd_spec(TaskID bwd_task_id, OpTaskSpec const &fwd_spec) const;
  OpTaskSpec infer_init_spec(TaskID init_task_id, OpTaskSpec const &bwd_spec) const;
  void infer_bwd_spec(OpTasksSpec &spec) const;
  void infer_init_spec(OpTasksSpec &spec) const;

  void execute_task_spec(FFModel const &, OpTaskSpec const &);
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

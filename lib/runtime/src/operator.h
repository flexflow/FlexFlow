#ifndef _FLEXFLOW_RUNTIME_SRC_OPERATOR_H
#define _FLEXFLOW_RUNTIME_SRC_OPERATOR_H

#include "runtime/config.h"
#include "layer_id.h"
#include "pcg/machine_view.h"
#include "parallel_tensor.h"
#include <vector>
#include "utils/stack_vector.h"
#include "tasks.h"
#include <stdexcept>
#include "task_spec.h"
#include "kernels/per_device_op_state.h"
#include "kernels/profiling.h"
#include "utils/strong_typedef.h"
#include "utils/stack_string.h"
#include "op-attrs/operator_attrs.h"

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_measure;
extern LegionRuntime::Logger::Category log_profile;

class Simulator;
class CostMetrics;
class FFModel;

struct op_guid_t : strong_typedef<op_guid_t, size_t> {
  using strong_typedef::strong_typedef;
};

class Op {
protected:
  void inner_measure_operator_cost(Simulator *sim,
                                   std::function<void(ffStream_t)> const &forward,
                                   std::function<void(ffStream_t)> const &backward,
                                   CostMetrics &cost_metrics) const;

public:
  Op(op_guid_t,
     OperatorType,
     DataType,
     std::string const &name,
     PCGOperatorAttrs const &attrs,
     bool profiling);
  
  // Pure virtual functions that must be implemented
  virtual void init(FFModel const &) = 0;
  virtual void forward(FFModel const &) = 0;
  virtual void backward(FFModel const &) = 0;

  virtual bool measure_operator_cost(Simulator *sim,
                                     MachineView const &mv,
                                     CostMetrics &cost_metrics) const = 0;
  virtual bool estimate_sync_cost(Simulator *sim,
                                  MachineView const &pc,
                                  CostMetrics &cost_metrics) const;
  //
  // Other virtual functions that can be optionally overwritten
  virtual Legion::Domain get_input_tensor_shape(MachineView const &pc,
                                                int input_idx,
                                                int part_idx) const; // deprecated
  virtual Legion::Domain get_output_tensor_shape(MachineView const &pc,
                                                 int output_idx,
                                                 int part_idx) const; // deprecated
  virtual Legion::Domain get_weight_tensor_shape(MachineView const &pc,
                                                 int weight_idx,
                                                 int part_idx) const; // deprecated
  virtual bool is_valid_parallel_config(FFModel const &ff,
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

  bool check_output_input_weight_same_parallel_is() const;

  void execute_task(FFModel const &, TaskID, OpTaskSignature const &);
  ParallelTensor const &get_parallel_tensor(TensorSpec const &) const;
  TensorSpec input_tensor(int idx) const;
  OpTaskBinding get_task_binding(OpTaskType) const;
  void set_argumentmap(OpTaskType, FFModel const &f, Legion::ArgumentMap);

  virtual OpTaskBinding get_init_task_binding() const = 0;
  virtual TaskID get_init_task_id() const = 0;
  virtual OpTaskBinding get_fwd_task_binding() const = 0;
  virtual TaskID get_fwd_task_id() const = 0;
  virtual OpTaskBinding get_bwd_task_binding() const = 0;
  virtual TaskID get_bwd_task_id() const = 0;
public:
  op_guid_t guid;
  OperatorType op_type;
  DataType data_type;
  // the guid of the layer associated with the current operator
  // layer_guid is used to match layer with op
  stack_string<MAX_OPNAME> name;
  /* Legion::IndexSpace parallel_is; */
  stack_vector<bool, MAX_NUM_INPUTS> trainableInputs;
  stack_vector<PerDeviceOpState *, MAX_NUM_WORKERS> meta;
  bool profiling;
#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
#endif
};

template <typename F, typename ...Ts, typename Str>
void profile(F const &f, bool profiling, Str s, Ts &&...ts) {
  optional<float> elapsed = profiling_wrapper<F, Ts...>(f, profiling, std::forward<Ts>(ts)...);
  if (elapsed.has_value()) {
    log_profile.debug(s, elapsed.value());
  }
}

template <TaskID TASK> OpTaskSignature get_signature();

OpTaskSignature get_signature(TaskID);

struct Operator {
public:
  Operator() = delete;

  Operator(op_guid_t,
           OperatorType, 
           DataType,
           std::string const &name,
           PCGOperatorAttrs const &attrs,
           bool profiling);

  Op *operator->();
  Op const *operator->() const;
private:
  std::shared_ptr<Op const> ptr;
};

struct OperatorManager {
public:
  OperatorManager() = default;

  Operator create(PCGOperatorAttrs const &attrs,
                  DataType data_type,
                  std::string const &name, 
                  bool profiling) {
    return {this->next_id(), get_op_type(attrs), data_type, name, attrs, profiling};
  }

  template <typename Variant>
  Operator create(Variant const &attrs, DataType data_type, std::string const &name, bool profiling) {
    return this->create(widen<PCGOperatorAttrs>(attrs), data_type, name, profiling);
  }

private:
  op_guid_t next_id() {
    return op_guid_t(this->op_global_guid++);
  }
private:
  size_t op_global_guid = OP_GUID_FIRST_VALID;
};

}

#endif 

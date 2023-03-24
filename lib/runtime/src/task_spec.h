#ifndef _FLEXFLOW_RUNTIME_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_TASK_SPEC_H

#include "legion.h"
#include "runtime/tasks.h"
#include "utils/optional.h"
#include "runtime/config.h"
#include <unordered_set>
#include <unordered_map>
#include "utils/bidict.h"
#include "accessor.h"
#include "serialization.h"
#include "ff_task_args.h"

namespace FlexFlow {

enum class TensorRole {
  INPUT,
  PARAM,
  OUTPUT,
};

enum class IsGrad {
  YES,
  NO
};

enum class IsTrainable {
  YES,
  NO
};

enum class OpTaskType {
  INIT,
  FWD,
  BWD
};

enum class SlotType {
  TENSOR,
  VARIADIC
};

using slot_id = int;

using region_idx = int;


struct TensorSpec {
  TensorSpec() = delete;
  TensorSpec(TensorRole, int, IsGrad is_grad = IsGrad::NO, optional<Legion::PrivilegeMode> mode = nullopt);

  TensorRole role;
  int idx;
  IsGrad is_grad;
  bool is_trainable;
  optional<Legion::PrivilegeMode> mode;

  TensorSpec grad() const;
};


Legion::PrivilegeMode get_default_mode(OpTaskType, TensorRole, IsGrad);

struct TensorSlotSpec {
  TensorSlotSpec() = delete;
  TensorSlotSpec(slot_id, SlotType, TensorRole, IsGrad, optional<Legion::PrivilegeMode> mode = nullopt);

  slot_id name;
  SlotType slot_type;
  TensorRole tensor_role;
  IsGrad is_grad;
  optional<Legion::PrivilegeMode> mode;

  Legion::PrivilegeMode get_privileges(OpTaskType) const;
};

TensorSlotSpec get_backward_slot(TensorSlotSpec const &forward_slot);
TensorSlotSpec get_backward_grad_slot(TensorSlotSpec const &forward_slot);

struct ArgSlotSpec {
  std::type_info type;
};

using SlotSpec = variant<TensorSlotSpec, ArgSlotSpec>;

bool is_tensor_slot(SlotSpec const &);
bool is_arg_slot(SlotSpec const &);
TensorSlotSpec get_tensor_slot(SlotSpec const &);
ArgSlotSpec get_arg_slot(SlotSpec const &);

struct OpTaskSignature {
  OpTaskSignature(OpTaskType);

  void add_slot(slot_id, TensorRole, SlotType);
  void add_grad_slot(slot_id, TensorRole, SlotType);

  void add_arg_slot(slot_id);

  void add_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_param_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_output_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_input_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_param_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_output_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  std::unordered_set<SlotSpec> get_slots() const;

private:
  std::vector<std::type_info> task_arg_types;
  std::unordered_map<int, TensorRole> slots;
};

struct OpTaskBinding {
  void bind(slot_id, TensorSpec const &);
  void bind_grad(slot_id, TensorSpec const &);

private:
  std::unordered_map<int, TensorSpec> bindings;

/*   tl::optional<TensorSpec const &> in_slot(slot_id) const; */
/*   std::vector<TensorSpec const &> in_variadic_slot(slot_id) const; */
};

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);

struct OpTaskArgumentFormat {
private:
  int region_idx_counter;
  bidict<TensorSpec, int> region_idxs;
};

OpTaskArgumentFormat generate_argument_format(OpTaskBinding const &);

struct OpTaskArgumentAccessor {
  OpTaskArgumentAccessor(Legion::Task const *task, 
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);

  template <typename T>
  T get_argument(slot_id);

  template <Legion::PrivilegeMode PRIV>
  typename privilege_mode_to_accessor<PRIV>::type get_tensor(slot_id);

private:
  Legion::Task const *task;
  std::vector<Legion::PhysicalRegion> const &regions;
  Legion::Context ctx;
  Legion::Runtime *runtime;
};


struct OpTaskSpec {
  OpTaskSpec(TaskID, OpTaskType);

  void bind(int, TensorSpec const &);
  void bind_grad(int, TensorSpec const &);

  void bind(std::vector<std::pair<int, TensorSpec>> const &);
  void bind_grad(std::vector<std::pair<int, TensorSpec>> const &);

  tl::optional<TensorSpec const &> in_slot(int) const;
  tl::optional<TensorSpec const &> in_slot_grad(int) const;
  int get_region_idx(TensorSpec const &) const;
  optional<int> get_region_idx(int slot) const;

  Legion::TaskArgument get_argument() const;

  bidict<TensorSpec, int> const &get_region_idxs() const;

  template <typename T>
  void add_arg(T const &t) {
    this->task_args.add_arg<T>(t);
  }

  template <typename T>
  T const *at(int idx, void *args) {
    return this->task_args.at<T>(idx, args);
  }

  template <typename T>
  T const *at(void *args) {
    return this->task_args.at<T>(args);
  }

  std::unordered_map<int, TensorRole> const &get_slots() const;
  std::unordered_map<int, TensorSpec> const &get_bindings() const;
private:
  int new_region_idx();

  int region_idx_counter = 0;
  OpTaskType task_type;
  FFTaskArgs task_args;
  std::unordered_map<int, TensorSpec> bindings;
  bidict<TensorSpec, int> region_idxs;
};

struct OpTasksSpec {
  OpTasksSpec(TaskID init, TaskID fwd, TaskID bwd);

  OpTaskSpec const &get_init() const;
  OpTaskSpec const &get_fwd() const;
  OpTaskSpec const &get_bwd() const;

  void set_init(OpTaskSpec const &);
  void set_fwd(OpTaskSpec const &); 
  void set_bwd(OpTaskSpec const &);

  TensorSpec input_tensor(int);
  TensorSpec output_tensor(int);

  OpTaskSpec const &get_task_spec(OpTaskType) const;

  bool is_defined(OpTaskType) const;
  bool is_fully_defined() const;
  TaskID get_task_id(OpTaskType) const;

private:
  variant<OpTaskSpec, TaskID> init_spec, fwd_spec, bwd_spec;
};

struct TaskAccessorGuide {
  std::unordered_map<slot_id, optional<std::pair<std::size_t, std::type_info>>> args;
  std::unordered_map<slot_id, optional<std::pair<region_idx, Legion::PrivilegeMode>>> regions;
};

struct TaskAccessor {
  TaskAccessor(Legion::Task const *task, 
               std::vector<Legion::PhysicalRegion> const &regions, 
               Legion::Context const &ctx, 
               Legion::Runtime *runtime, 
               TaskAccessorGuide);

  TaskAccessor(Legion::Task const *task, 
               std::vector<Legion::PhysicalRegion> const &regions, 
               Legion::Context const &ctx, Legion::Runtime *runtime, 
               OpTaskType task_type);

  template <typename DT>
  DT *get_slot(int slot) const {
    optional<TensorSpec> maybe_tensor = this->spec.in_slot(slot);
    if (!maybe_tensor.has_value()) {
      return nullptr;
    }
    TensorSpec tensor = maybe_tensor.value();
    assert (tensor.mode == READ_ONLY || tensor.mode == READ_WRITE || tensor.mode == WRITE_ONLY);

    int region_idx = this->spec.get_region_idx(tensor);
    assert (this->regions.size() > region_idx);
    assert (this->task->regions.size() > region_idx);

    if (tensor.mode == READ_ONLY) {
      throw std::runtime_error("Cannot access readonly tensor as non-const");
    } else if (tensor.mode == READ_WRITE) {
      return helperGetTensorPointerRW<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime);
    } else if (tensor.mode == WRITE_ONLY) {
      return helperGetTensorPointerWO<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime);
    }
  }

  template <typename DT>
  DT const *get_const_slot(int slot) const {
    optional<TensorSpec> maybe_tensor = this->spec.in_slot(slot);
    if (!maybe_tensor.has_value()) {
      return nullptr;
    }
    TensorSpec tensor = maybe_tensor.value();
    assert (tensor.mode == READ_ONLY || tensor.mode == READ_WRITE || tensor.mode == WRITE_ONLY);

    int region_idx = this->spec.get_region_idx(tensor);
    assert (this->regions.size() > region_idx);
    assert (this->task->regions.size() > region_idx);

    if (tensor.mode == READ_ONLY) {
      return helperGetTensorPointerRO<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime);
    } else if (tensor.mode == READ_WRITE) {
      return helperGetTensorPointerRW<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime);
    } else if (tensor.mode == WRITE_ONLY) {
      throw std::runtime_error("Cannot access writeonly tensor as const");
    } 
  }

  template <typename DT>
  DT *get_slot_grad(int slot) const {

  }

  template <typename DT>
  DT const *get_const_slot_grad(int slot) const {

  }


  template <typename T>
  T const *get_arg(int idx) {
    return this->spec.at<T>(idx, task->args);
  }

  template <typename T>
  T const *get_arg() {
    return this->spec.at<T>(task->args);
  }

private:
  Legion::Task const *task;
  std::vector<Legion::PhysicalRegion> const &regions;
  Legion::Context const &ctx;
  Legion::Runtime *runtime;
  OpTaskSpec spec;
};



}

VISITABLE_STRUCT(::FlexFlow::TaskAccessorGuide, args, regions);


#endif

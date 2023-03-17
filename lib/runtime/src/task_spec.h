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

struct OpTaskSignature {
  OpTaskSignature(TaskID, OpTaskType);

  void add_slot(int, TensorRole);
  void add_grad_slot(int, TensorRole);

  void add_input_slot(int);
  void add_param_slot(int);
  void add_output_slot(int);
  void add_input_grad_slot(int);
  void add_param_grad_slot(int);
  void add_output_grad_slot(int);

  template <typename T>
  void add_arg_slot(int);

  TaskID get_task_id() const;
private:
  TaskID task_id;

  std::vector<std::type_info> task_arg_types;
  std::unordered_map<int, TensorRole> slots;
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

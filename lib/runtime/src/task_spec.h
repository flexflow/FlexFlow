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
#include <typeindex>
#include "utils/stack_map.h"
#include "accessor.h"

namespace FlexFlow {

struct Op;

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
  TensorSpec(TensorRole, int, bool is_trainable, DataType, IsGrad is_grad = IsGrad::NO, optional<Legion::PrivilegeMode> mode = nullopt);
  TensorSpec(TensorRole, int, IsTrainable, DataType, IsGrad is_grad = IsGrad::NO, optional<Legion::PrivilegeMode> mode = nullopt);

  bool operator==(TensorSpec const &) const;
  bool operator!=(TensorSpec const &) const;

  TensorRole role;
  int idx;
  IsGrad is_grad;
  IsTrainable is_trainable;
  DataType datatype;
  optional<Legion::PrivilegeMode> mode;

  TensorSpec grad() const;

  Legion::PrivilegeMode get_privileges() const;
};

TensorSpec input_tensor(int, IsTrainable);
TensorSpec input_tensor(int, bool);

TensorSpec output_tensor(int);
TensorSpec param_tensor(int);

Legion::PrivilegeMode get_default_mode(OpTaskType, TensorRole, IsGrad);

struct TensorSlotSpec {
  TensorSlotSpec() = delete;
  TensorSlotSpec(slot_id, SlotType, TensorRole, IsGrad);

  slot_id name;
  SlotType slot_type;
  TensorRole tensor_role;
  IsGrad is_grad;

  Legion::PrivilegeMode get_privileges(OpTaskType) const;
};

TensorSlotSpec get_backward_slot(TensorSlotSpec const &forward_slot);
TensorSlotSpec get_backward_grad_slot(TensorSlotSpec const &forward_slot);

using ArgSlotSpec = std::type_index;

struct ArgSpec {
  std::type_index type;
  size_t start;
  size_t size;

  size_t stop() const;
};

using SlotSpec = variant<TensorSlotSpec, ArgSlotSpec>;

bool is_tensor_slot(SlotSpec const &);
bool is_arg_slot(SlotSpec const &);
TensorSlotSpec get_tensor_slot(SlotSpec const &);
ArgSlotSpec get_arg_slot(SlotSpec const &);

struct OpTaskSignature {
  OpTaskSignature(OpTaskType);

  void add_slot(SlotSpec const &);
  void add_slot(TensorSlotSpec const &);

  void add_slot(slot_id, TensorRole, SlotType);
  void add_grad_slot(slot_id, TensorRole, SlotType);

  template <typename T>
  void add_arg_slot(slot_id name) {
    static_assert(is_serializable<T>, "Type must be serializable");

    this->task_arg_types.insert({ name, typeid(T) });
  }

  void add_input_slot(slot_id);
  void add_input_slot(slot_id, SlotType);
  void add_input_slot(slot_id, Legion::PrivilegeMode);
  void add_input_slot(slot_id, SlotType, Legion::PrivilegeMode);

  void add_param_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_output_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_input_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_param_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_output_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  bool operator==(OpTaskSignature const &) const;

  std::unordered_set<SlotSpec> get_slots() const;

  SlotSpec get_slot(slot_id) const;
  OpTaskType get_task_type() const;
private:
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_map<slot_id, TensorRole> slots;
};

using TensorArgumentFormat = variant<
  std::pair<region_idx, TensorSpec>,
  std::vector<std::pair<region_idx, TensorSpec>>
>;

using SlotKey = std::pair<slot_id, IsGrad>;

constexpr size_t MAX_NUM_TASK_REGIONS = 20;
constexpr size_t MAX_NUM_TASK_ARGUMENTS = 5;

struct OpTaskArgumentFormat {
  stack_map<SlotKey, TensorArgumentFormat, MAX_NUM_TASK_REGIONS> region_idxs;
  stack_map<slot_id, ArgSpec, MAX_NUM_TASK_ARGUMENTS> argument_offsets;
  ArgSpec self_offset;
};

struct OpTaskBinding {
  OpTaskBinding() {
    serializer.reserve_bytes(sizeof(OpTaskArgumentFormat));
  }

  void bind(slot_id, TensorSpec const &);
  void bind_grad(slot_id, TensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    auto arg_spec = this->generate_arg_spec<T>(t);
    assert (!contains_key(this->arg_bindings, name));
    arg_bindings.insert({name, arg_spec});
  }

  void bind(std::vector<std::pair<slot_id, TensorSpec>> const &);

  std::unordered_map<slot_id, TensorSpec> const &get_tensor_bindings() const;
  std::unordered_map<slot_id, ArgSpec> const &get_arg_bindings() const;

  Legion::TaskArgument get_legion_task_arg() const;
private:
  template <typename T>
  ArgSpec generate_arg_spec(T const &t) {
    static_assert(is_serializable<T>, "Type must be serializable");

    size_t pre_size = serializer.get_used_bytes();
    ff_task_serialize(serializer, t);
    size_t post_size = serializer.get_used_bytes();
    return {
      typeid(T),
      pre_size,
      post_size - pre_size
    };
  }

  Legion::Serializer serializer;
  std::unordered_map<slot_id, ArgSpec> arg_bindings;
  std::unordered_map<slot_id, TensorSpec> bindings;
  void *task_format_location;

  friend OpTaskArgumentFormat compile_task_invocation(OpTaskSignature const &, OpTaskBinding &);
};

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

std::unordered_map<int, TensorSpec> get_regions_idxs(OpTaskArgumentFormat const &);

OpTaskArgumentFormat compile_task_invocation(OpTaskSignature const &, OpTaskBinding const &);

struct OpTaskArgumentAccessor {
  OpTaskArgumentAccessor(Legion::Task const *task, 
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);

  template <typename T>
  T const &get_argument(slot_id slot) {
    ArgSpec arg_spec = this->args_fmt.argument_offsets.at(slot);

    std::type_index requested_type = {typeid(T)};
    if (arg_spec.type != requested_type) {
      std::ostringstream oss;
      oss << "Type mismatch in argument access: \"" << arg_spec.type.name() << "\" != \"" << requested_type.name() << "\"";
      throw std::runtime_error(oss.str());
    }

    void *start_ptr = &((std::uint8_t*)this->task->args)[arg_spec.start];
    Legion::Deserializer dez(start_ptr, arg_spec.size);

    return ff_task_deserialize<T>(dez);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_generic_accessor(TensorSpec const &tensor_spec, region_idx idx) {
    auto tensor_privs = tensor_spec.get_privileges();
    if (tensor_privs != PRIV) {
      std::ostringstream oss;
      oss << "Privilege mismatch while accessing tensor: " << to_string(tensor_privs) << " != " << to_string(PRIV);
      throw std::runtime_error(oss.str());
    }
    
    return helperGetGenericTensorAccessor<PRIV>(tensor_spec.datatype, regions[idx], task->regions[idx], FID_DATA, ctx, runtime);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_generic_accessor(std::pair<region_idx, TensorSpec> const &p) {
    return this->get_generic_accessor<PRIV>(p.second, p.first);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot, IsGrad is_grad) {
    auto argument_format = get<std::pair<int, TensorSpec>>(this->args_fmt.region_idxs.at({slot, is_grad}));

    return this->get_generic_accessor<PRIV>(argument_format);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot) {
    return this->get_tensor<PRIV>(slot, IsGrad::NO);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(slot_id slot) {
    return this->get_tensor<PRIV>(slot, IsGrad::YES);
  }

  template <Legion::PrivilegeMode PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>> get_variadic_tensor(slot_id slot, IsGrad is_grad) {
    std::vector<privilege_mode_to_accessor<PRIV>> result;

    auto argument_format = get<std::vector<std::pair<region_idx, TensorSpec>>>(this->args_fmt.region_idxs.at({slot, is_grad}));
    for (auto const &argument : argument_format) {
      result.push_back(this->get_generic_accessor<PRIV>(argument));
    }

    return result;
  }

  template <Legion::PrivilegeMode PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>> get_variadic_tensor(slot_id slot) {
    return this->get_variadic_tensor<PRIV>(slot, IsGrad::NO);
  }

  template <Legion::PrivilegeMode PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>> get_variadic_tensor_grad(slot_id slot) {
    return this->get_variadic_tensor<PRIV>(slot, IsGrad::YES); 
  }
private:
  Legion::Task const *task;
  std::vector<Legion::PhysicalRegion> const &regions;
  Legion::Context ctx;
  Legion::Runtime *runtime;
  OpTaskArgumentFormat const &args_fmt;
};


/* struct TaskAccessorGuide { */
/*   std::unordered_map<slot_id, optional<std::pair<std::size_t, std::type_info>>> args; */
/*   std::unordered_map<slot_id, optional<std::pair<region_idx, Legion::PrivilegeMode>>> regions; */
/* }; */

/* struct OpTaskSpec { */
/*   OpTaskSpec(TaskID, OpTaskType); */

/*   void bind(int, TensorSpec const &); */
/*   void bind_grad(int, TensorSpec const &); */

/*   void bind(std::vector<std::pair<int, TensorSpec>> const &); */
/*   void bind_grad(std::vector<std::pair<int, TensorSpec>> const &); */

/*   tl::optional<TensorSpec const &> in_slot(int) const; */
/*   tl::optional<TensorSpec const &> in_slot_grad(int) const; */
/*   int get_region_idx(TensorSpec const &) const; */
/*   optional<int> get_region_idx(int slot) const; */

/*   Legion::TaskArgument get_argument() const; */

/*   bidict<TensorSpec, int> const &get_region_idxs() const; */

/*   template <typename T> */
/*   void add_arg(T const &t) { */
/*     this->task_args.add_arg<T>(t); */
/*   } */

/*   template <typename T> */
/*   T const *at(int idx, void *args) { */
/*     return this->task_args.at<T>(idx, args); */
/*   } */

/*   template <typename T> */
/*   T const *at(void *args) { */
/*     return this->task_args.at<T>(args); */
/*   } */

/*   std::unordered_map<int, TensorRole> const &get_slots() const; */
/*   std::unordered_map<int, TensorSpec> const &get_bindings() const; */
/* private: */
/*   int new_region_idx(); */

/*   int region_idx_counter = 0; */
/*   OpTaskType task_type; */
/*   FFTaskArgs task_args; */
/*   std::unordered_map<int, TensorSpec> bindings; */
/*   bidict<TensorSpec, int> region_idxs; */
/* }; */

/* struct OpTasksSpec { */
/*   OpTasksSpec(TaskID init, TaskID fwd, TaskID bwd); */

/*   OpTaskSpec const &get_init() const; */
/*   OpTaskSpec const &get_fwd() const; */
/*   OpTaskSpec const &get_bwd() const; */

/*   void set_init(OpTaskSpec const &); */
/*   void set_fwd(OpTaskSpec const &); */ 
/*   void set_bwd(OpTaskSpec const &); */

/*   TensorSpec input_tensor(int); */
/*   TensorSpec output_tensor(int); */

/*   OpTaskSpec const &get_task_spec(OpTaskType) const; */

/*   bool is_defined(OpTaskType) const; */
/*   bool is_fully_defined() const; */
/*   TaskID get_task_id(OpTaskType) const; */

/* private: */
/*   variant<OpTaskSpec, TaskID> init_spec, fwd_spec, bwd_spec; */
/* }; */

/* struct TaskAccessor { TaskAccessor(Legion::Task const *task, std::vector<Legion::PhysicalRegion> const &regions, Legion::Context const &ctx, Legion::Runtime *runtime, TaskAccessorGuide); */

/*   TaskAccessor(Legion::Task const *task, */ 
/*                std::vector<Legion::PhysicalRegion> const &regions, */ 
/*                Legion::Context const &ctx, Legion::Runtime *runtime, */ 
/*                OpTaskType task_type); */

/*   template <typename DT> */
/*   DT *get_slot(int slot) const { */
/*     optional<TensorSpec> maybe_tensor = this->spec.in_slot(slot); */
/*     if (!maybe_tensor.has_value()) { */
/*       return nullptr; */
/*     } */
/*     TensorSpec tensor = maybe_tensor.value(); */
/*     assert (tensor.mode == READ_ONLY || tensor.mode == READ_WRITE || tensor.mode == WRITE_ONLY); */

/*     int region_idx = this->spec.get_region_idx(tensor); */
/*     assert (this->regions.size() > region_idx); */
/*     assert (this->task->regions.size() > region_idx); */

/*     if (tensor.mode == READ_ONLY) { */
/*       throw std::runtime_error("Cannot access readonly tensor as non-const"); */
/*     } else if (tensor.mode == READ_WRITE) { */
/*       return helperGetTensorPointerRW<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime); */
/*     } else if (tensor.mode == WRITE_ONLY) { */
/*       return helperGetTensorPointerWO<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime); */
/*     } */
/*   } */

/*   template <typename DT> */
/*   DT const *get_const_slot(int slot) const { */
/*     optional<TensorSpec> maybe_tensor = this->spec.in_slot(slot); */
/*     if (!maybe_tensor.has_value()) { */
/*       return nullptr; */
/*     } */
/*     TensorSpec tensor = maybe_tensor.value(); */
/*     assert (tensor.mode == READ_ONLY || tensor.mode == READ_WRITE || tensor.mode == WRITE_ONLY); */

/*     int region_idx = this->spec.get_region_idx(tensor); */
/*     assert (this->regions.size() > region_idx); */
/*     assert (this->task->regions.size() > region_idx); */

/*     if (tensor.mode == READ_ONLY) { */
/*       return helperGetTensorPointerRO<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime); */
/*     } else if (tensor.mode == READ_WRITE) { */
/*       return helperGetTensorPointerRW<DT>(regions[region_idx], task->regions[region_idx], FID_DATA, ctx, runtime); */
/*     } else if (tensor.mode == WRITE_ONLY) { */
/*       throw std::runtime_error("Cannot access writeonly tensor as const"); */
/*     } */ 
/*   } */

/*   template <typename DT> */
/*   DT *get_slot_grad(int slot) const { */

/*   } */

/*   template <typename DT> */
/*   DT const *get_const_slot_grad(int slot) const { */

/*   } */


/*   template <typename T> */
/*   T const *get_arg(int idx) { */
/*     return this->spec.at<T>(idx, task->args); */
/*   } */

/*   template <typename T> */
/*   T const *get_arg() { */
/*     return this->spec.at<T>(task->args); */
/*   } */

/* private: */
/*   Legion::Task const *task; */
/*   std::vector<Legion::PhysicalRegion> const &regions; */
/*   Legion::Context const &ctx; */
/*   Legion::Runtime *runtime; */
/*   OpTaskSpec spec; */
/* }; */

}

VISITABLE_STRUCT(::FlexFlow::TensorSpec, role, idx, is_grad, is_trainable, mode);


#endif

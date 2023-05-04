#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H

#include "utils/strong_typedef.h"
#include "utils/visitable.h"
#include <typeindex>
#include "serialization.h"
#include "accessor.h"
#include "runtime/config.h"
#include "parallel_tensor_guid_t.h"
#include "tasks.h"

namespace FlexFlow {

struct RuntimeBacking;

enum class InvocationType {
  INDEX,
  STANDARD
};

enum class SlotType {
  TENSOR,
  VARIADIC
};

enum class ArgSlotType {
  INDEX,
  STANDARD
};

enum class IsGrad {
  YES,
  NO
};

struct ArgSpec : public use_visitable_cmp<ArgSpec> {
public:
  ArgSpec() = delete;
  ArgSpec(std::type_index, size_t start, size_t size);

  size_t get_stop() const;
public:
  std::type_index type;
  size_t start;
  size_t size;
};

/* struct IndexArgSpec : public use_visitable_cmp<IndexArgSpec> { */

/* }; */

struct slot_id : strong_typedef<slot_id, int> {
  using strong_typedef::strong_typedef;

  slot_id(int);
};

struct region_idx_t : strong_typedef<region_idx_t, int> {
  using strong_typedef::strong_typedef;
};

}

MAKE_TYPEDEF_HASHABLE(::FlexFlow::slot_id);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::slot_id, "slot_id");

MAKE_TYPEDEF_HASHABLE(::FlexFlow::region_idx_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::region_idx_t, "region_idx");

namespace FlexFlow {

struct ParallelTensorSpec : public use_visitable_cmp<ParallelTensorSpec> {
public:
  ParallelTensorSpec() = delete;
  ParallelTensorSpec(parallel_tensor_guid_t);

  ParallelTensorSpec grad() const;
public:
  parallel_tensor_guid_t parallel_tensor_guid; 
  IsGrad is_grad = IsGrad::NO;
};

struct ParallelTensorSlotSpec {
public:
  ParallelTensorSlotSpec() = delete;
  ParallelTensorSlotSpec(SlotType, Legion::PrivilegeMode);

public:
  SlotType slot_type;
  Legion::PrivilegeMode privileges;
};

template <typename T> 
struct TypedFuture {
  T get();
};

template <typename T> 
struct TypedFutureMap {
  T get(Legion::DomainPoint);
};

struct TaskSignature {
  TaskSignature() = default;

  void add_slot(slot_id, ParallelTensorSlotSpec const &);

  template <typename T>
  void add_arg_slot(slot_id name) {
    static_assert(is_serializable<T>, "Argument type must be serializable");

    this->task_arg_types.insert({ name, { typeid(T) }});
  }

  template <typename T>
  void add_return_value();

  template <typename T>
  void add_variadic_arg_slot(slot_id name);

  /* template <typename T, typename F> */
  /* void add_index_arg_slot(slot_id name, F const &idx_to_arg) { */
  /*   static_assert(is_serializable<T>, "Argument type must be serializable"); */

  /*   this->task_arg_types.insert({ name, { typeid(T), ArgSlotType::INDEX }}); */
  /* } */

  bool operator==(TaskSignature const &) const;
  bool operator!=(TaskSignature const &) const;
private:
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_map<slot_id, ParallelTensorSlotSpec> tensor_slots;
};

struct TaskBinding {
public:
  TaskBinding(optional<Legion::Domain const &> = nullopt);
  TaskBinding(InvocationType);

  void bind(slot_id, ParallelTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    auto arg_spec = this->generate_arg_spec<T>(t);
    assert (!contains_key(this->arg_bindings, name));
    arg_bindings.insert({name, arg_spec});
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &f) {
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &f) {
  }

  template <typename F, typename T = decltype(std::declval<F>()(std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f) {
    assert (this->domain.has_value());
    Legion::Domain d = this->domain.value();
    assert (!contains_key(this->arg_bindings, name));
    for (Legion::Domain::DomainPointIterator it(d); it; it++) {
      auto arg_spec = this->generate_idx_arg_spec<T>(f(*it), *it);
      arg_bindings.insert({{*it, name}, arg_spec});
    }
  }
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

  template <typename T>
  ArgSpec generate_idx_arg_spec(T const &t, Legion::DomainPoint const &pt) {
    static_assert(is_serializable<T>, "Type must be serializable");

    Legion::Serializer ser = this->idx_serializers.at(pt);
    size_t pre_size = ser.get_used_bytes();
    ff_task_serialize(ser, t);
    size_t post_size = ser.get_used_bytes();
    return {
      typeid(T), 
      pre_size, 
      post_size - pre_size
    };
  }

private:
  Legion::Serializer serializer;
  optional<Legion::Domain> domain;
  std::unordered_map<slot_id, ArgSpec> arg_bindings;
  std::map<Legion::DomainPoint, Legion::Serializer> idx_serializers;
  std::unordered_map<std::pair<slot_id, Legion::DomainPoint>, ArgSpec> index_arg_bindings;
  std::unordered_map<slot_id, ParallelTensorSpec> bindings;
};

using NonvariadicFormat = std::pair<region_idx_t, ParallelTensorSpec>;
using VariadicFormat = std::vector<NonvariadicFormat>;

using TensorArgumentFormat = variant<NonvariadicFormat, VariadicFormat>;

bool is_variadic(TensorArgumentFormat const &);
VariadicFormat get_variadic_format(TensorArgumentFormat const &);
NonvariadicFormat get_nonvariadic_format(TensorArgumentFormat const &);

struct TaskArgumentFormat : use_visitable_eq<TaskArgumentFormat> {
  stack_map<slot_id, TensorArgumentFormat, MAX_NUM_TASK_REGIONS> region_idxs;
  stack_map<slot_id, ArgSpec, MAX_NUM_TASK_ARGUMENTS> argument_offsets;
  stack_map<region_idx_t, Legion::PrivilegeMode, MAX_NUM_TASK_REGIONS> regions;
  stack_map<ParallelTensorSpec, DataType, MAX_NUM_TASK_REGIONS> data_types;
  ArgSpec self_offset;
};

Legion::PrivilegeMode get_privileges(TaskArgumentFormat const &, region_idx_t);
Legion::PrivilegeMode get_privileges(TaskArgumentFormat const &, ParallelTensorSpec const &);
region_idx_t get_region_idx(TaskArgumentFormat const &, ParallelTensorSpec const &);
DataType get_datatype(TaskArgumentFormat const &, ParallelTensorSpec const &);

struct TaskArgumentAccessor {
  TaskArgumentAccessor(Legion::Task const *task, 
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

  template <typename T>
  optional<T> get_optional_argument(slot_id);

  template <typename T>
  std::vector<T> get_variadic_argument(slot_id);

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_generic_accessor(ParallelTensorSpec const &tensor_spec, region_idx_t idx) {
    auto tensor_privs = get_privileges(this->args_fmt, tensor_spec);
    if (tensor_privs != PRIV) {
      std::ostringstream oss;
      oss << "Privilege mismatch while accessing tensor: " << to_string(tensor_privs) << " != " << to_string(PRIV);
      throw std::runtime_error(oss.str());
    }
    
    return helperGetGenericTensorAccessor<PRIV>(get_datatype(this->args_fmt, tensor_spec), regions[idx.value()], task->regions[idx.value()], FID_DATA, ctx, runtime);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_generic_accessor(std::pair<region_idx_t, ParallelTensorSpec> const &p) {
    return this->get_generic_accessor<PRIV>(p.second, p.first);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id slot) {
    auto argument_format = get<NonvariadicFormat>(this->args_fmt.region_idxs.at(slot));

    return this->get_generic_accessor<PRIV>(argument_format);
  }

  template <Legion::PrivilegeMode PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(slot_id slot) {
    return this->get_tensor<PRIV>(slot, IsGrad::YES);
  }

  template <Legion::PrivilegeMode PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>> get_variadic_tensor(slot_id slot) {
    std::vector<privilege_mode_to_accessor<PRIV>> result;

    auto argument_format = get<VariadicFormat>(this->args_fmt.region_idxs.at(slot));
    for (auto const &argument : argument_format) {
      result.push_back(this->get_generic_accessor<PRIV>(argument));
    }

    return result;
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
  TaskArgumentFormat const &args_fmt;
};

struct TaskInvocation : public use_visitable_cmp<TaskInvocation> {
public:
  TaskInvocation() = delete;
  TaskInvocation(task_id_t const &task_id, TaskBinding const &binding)
    : task_id(task_id), binding(binding) { }

public:
  task_id_t task_id;
  TaskBinding binding;
};

/* TaskArgumentFormat compile_task_invocation(TaskInvocation const &); */

/* std::unordered_map<Legion::DomainPoint, TaskArgumentFormat> compile_index_task_invocation(TaskSignature const &signature, */
/*                                                                                           TaskBinding const &binding); */

struct TaskReturnAccessor { 
  template <typename T>
  TypedFuture<T> get_returned_future();

  template <typename T>
  TypedFutureMap<T> get_returned_future_map();
};

TaskReturnAccessor execute_task(LegionConfig const &config, 
                                TaskInvocation const &,
                                RuntimeBacking const &backing);

template <task_id_t> TaskSignature get_signature();
TaskSignature get_signature(task_id_t);

template <typename F>
void register_task(task_id_t, std::string const &name, TaskSignature const &, F const &func);

template <typename F>
void register_task(task_id_t, std::string const &name, TaskSignature const &, F const &func, F const &cpu_func); 

}

VISITABLE_STRUCT(::FlexFlow::TaskInvocation, task_id, binding);

#endif

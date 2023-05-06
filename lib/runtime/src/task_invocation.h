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
#include "task_signature.h"

namespace FlexFlow {

struct RuntimeBacking;

enum class InvocationType {
  INDEX,
  STANDARD
};

enum class ArgSlotType {
  INDEX,
  STANDARD
};

enum class IsGrad {
  YES,
  NO
};

struct ConcreteArgSpec {
public:
  ConcreteArgSpec() = delete;

  template <typename T>
  T const &get() {
    assert (std::type_index(typeid(T)) == this->type);

    return *(T const *)ptr.get();
  }

  template <typename T>
  static ConcreteArgSpec create(T const &t) {
    static_assert(is_serializable<T>, "Type must be serializable");

    return ConcreteArgSpec(std::type_index(typeid(T)), std::make_shared<T>(t));
  }
private:
  ConcreteArgSpec(std::type_index, std::shared_ptr<void const *>);

  std::type_index type;
  std::shared_ptr<void const *> ptr;
};

// struct ArgSpec : public use_visitable_cmp<ArgSpec> {
// public:
//   ArgSpec() = delete;
//   ArgSpec(std::type_index, size_t start, size_t size);
// 
//   size_t get_stop() const;
// public:
//   std::type_index type;
//   size_t start;
//   size_t size;
// };

/* struct IndexArgSpec : public use_visitable_cmp<IndexArgSpec> { */

/* }; */

}

namespace FlexFlow {

struct ParallelTensorSpec : public use_visitable_cmp<ParallelTensorSpec> {
public:
  ParallelTensorSpec() = delete;
  explicit ParallelTensorSpec(parallel_tensor_guid_t, IsGrad is_grad = IsGrad::NO);

  ParallelTensorSpec grad() const;
public:
  parallel_tensor_guid_t parallel_tensor_guid; 
  IsGrad is_grad;
};

template <typename T> 
struct TypedFuture {
  explicit TypedFuture(Legion::Future const &future)
    : future(future)
  { }

  T get() {
    return future.get_result<T>();
  }
public:
  Legion::Future future;
};

struct CheckedTypedFuture {
public:
  CheckedTypedFuture() = delete;

  template <typename T>
  TypedFuture<T> get() {
    assert (std::type_index(typeid(T)) == this->type);

    return this->future;
  }

  template <typename T>
  static 
  CheckedTypedFuture create(TypedFuture<T> const &f) {
    return CheckedTypedFuture(std::type_index(typeid(T)), f);
  }
private:
  CheckedTypedFuture(std::type_index, Legion::Future const &);

  std::type_index type;
  Legion::Future future;
};

template <typename T> 
struct TypedFutureMap {
  explicit TypedFutureMap(Legion::FutureMap const &future_map)
    : future_map(future_map)
  { }

  T get(Legion::DomainPoint const &p) {
    return future_map.get_result<T>(p);
  }
public:
  Legion::FutureMap future_map;
};

struct CheckedTypedFutureMap {
public:
  CheckedTypedFutureMap() = delete;

  template <typename T>
  TypedFutureMap<T> get() {
    assert (std::type_index(typeid(T)) == this->type);

    return { this->future_map };
  }

  template <typename T>
  static 
  CheckedTypedFutureMap create(TypedFutureMap<T> const &fm) {
    return CheckedTypedFuturemap(std::type_index(typeid(T)), fm);
  }
public:
  CheckedTypedFutureMap(std::type_index, Legion::FutureMap const &);

  std::type_index type;
  Legion::FutureMap future_map;
};

using ArgSpec = variant<ConcreteArgSpec, CheckedTypedFuture, CheckedTypedFutureMap>;

struct TaskBinding {
public:
  explicit TaskBinding(optional<Legion::Domain const &> = nullopt);
  explicit TaskBinding(InvocationType);

  void bind(slot_id, ParallelTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    this->insert_arg_spec(name, ConcreteArgSpec::create(t));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &f) {
    this->insert_arg_spec(name, CheckedTypedFuture::create(f));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &fm) {
    this->insert_arg_spec(name, CheckedTypedFutureMap::create(fm));
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
  void insert_arg_spec(slot_id name, ArgSpec const &arg_spec) {
    assert (!contains_key(this->arg_bindings, name));
    arg_bindings.insert({ name, arg_spec });
  }

private:
  optional<Legion::Domain> domain;
  std::unordered_map<slot_id, ArgSpec> arg_bindings;
  std::map<Legion::DomainPoint, Legion::Serializer> idx_serializers;
  std::unordered_map<std::pair<slot_id, Legion::DomainPoint>, ArgSpec> index_arg_bindings;
  std::unordered_map<slot_id, ParallelTensorSpec> bindings;
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

TaskReturnAccessor execute_task(LegionConfig const &config, 
                                TaskInvocation const &,
                                RuntimeBacking const &backing);


}

VISITABLE_STRUCT(::FlexFlow::TaskInvocation, task_id, binding);

#endif

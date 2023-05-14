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
#include "profiling.h"
#include "utils/variant.h"

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
  ConcreteArgSpec(std::type_index, std::shared_ptr<void const>);

  std::type_index type;
  std::shared_ptr<void const *> ptr;
};

template <typename T>
struct IndexArg {
  IndexArg() = delete;

  template <typename F>
  IndexArg(F const &f) 
    : f(f) {
    static_assert(std::is_same<decltype(std::declval<F>()(std::declval<Legion::DomainPoint>())), T>::value, "");
  }

  T get(Legion::DomainPoint const &p) {
    return f(p);
  }
private:
  std::function<T(Legion::DomainPoint const &)> f;
};

struct IndexArgSpec {
public:
  template <typename T>
  T get(Legion::DomainPoint const &p) {
    assert (std::type_index(typeid(T)) == this->return_type);

    return *(T const *)(f(p).get());
  }

  template <typename F, typename T = decltype(std::declval<F>()(std::declval<Legion::DomainPoint>()))>
  static IndexArgSpec create(F const &ff) {
    static_assert(is_serializable<T>, "Type must be serializable");

    std::function<std::shared_ptr<void>(Legion::DomainPoint const &)> wrapped = [=](Legion::DomainPoint const &p) {
      return std::make_shared<T>(ff(p));
    };

    return IndexArgSpec(std::type_index(typeid(T)), wrapped);
  }
private:
  IndexArgSpec(std::type_index, std::function<std::shared_ptr<void>(Legion::DomainPoint const &)> const &);

  std::type_index return_type;
  std::function<std::shared_ptr<void>(Legion::DomainPoint const &)> f;
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

struct ParallelTensorSpec : public use_visitable_cmp<ParallelTensorSpec> {
public:
  ParallelTensorSpec() = delete;
  ParallelTensorSpec(parallel_tensor_guid_t, IsGrad is_grad = IsGrad::NO);

  ParallelTensorSpec grad() const;
public:
  parallel_tensor_guid_t parallel_tensor_guid; 
  IsGrad is_grad;
};

enum class ArgRefType {
  ENABLE_PROFILING,
  FF_HANDLE,
};

template <typename T>
struct ArgRef : public use_visitable_cmp<ArgRef<T>> {
public:
  ArgRef() = delete;
  ArgRef(ArgRefType ref_type)
    : ref_type(ref_type)
  { }

public:
  ArgRefType ref_type;
};

struct ArgRefSpec {
};


ArgRef<EnableProfiling> enable_profiling();
ArgRef<PerDeviceFFHandle> ff_handle();

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
    return CheckedTypedFuture(std::type_index(typeid(T)), f.future);
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
    return CheckedTypedFutureMap(std::type_index(typeid(T)), fm.future_map);
  }
public:
  CheckedTypedFutureMap(std::type_index, Legion::FutureMap const &);

  std::type_index type;
  Legion::FutureMap future_map;
};

using ArgSpec = variant<ConcreteArgSpec, IndexArgSpec, CheckedTypedFuture, CheckedTypedFutureMap, ArgRefSpec>;

template <typename T>
using TypedTaskArg = variant<T, IndexArg<T>, TypedFuture<T>, TypedFutureMap<T>, ArgRef<T>>;

std::type_index get_type_index(ArgSpec);

struct TaskInvocation;

template <typename T> struct TypedTaskInvocation { };

template <typename T>
TypedTaskInvocation<T> check_return_type(TaskInvocation const &);

struct TaskBinding {
public:
  static TaskBinding index_launch(MachineView const &);
  static TaskBinding index_launch(parallel_tensor_guid_t const &);
  static TaskBinding index_launch(slot_id const &);
  static TaskBinding standard_launch();
  static TaskBinding sync_type_dependent_launch(parallel_tensor_guid_t);
  static TaskBinding sync_type_dependent_launch(slot_id);

  void bind(slot_id, parallel_tensor_guid_t const &);

  template <typename T>
  void bind_arg(slot_id name, TypedTaskArg<T> const &);

  template <typename T>
  void bind_arg(slot_id, TypedTaskInvocation<T> const &);

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
    this->insert_arg_spec(name, IndexArgSpec::create(f));
  }
private:
  void insert_arg_spec(slot_id name, ArgSpec const &arg_spec) {
    assert (!contains_key(this->arg_bindings, name));
    arg_bindings.insert({ name, arg_spec });
  }

private:
  std::unordered_map<slot_id, ArgSpec> arg_bindings;
  std::unordered_map<slot_id, parallel_tensor_guid_t> bindings;
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


}

VISITABLE_STRUCT(::FlexFlow::TaskInvocation, task_id, binding);

#endif

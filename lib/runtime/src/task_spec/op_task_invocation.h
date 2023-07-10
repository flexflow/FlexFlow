#ifndef _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H

#include "accessor.h"
#include "index_task_invocation.h"
#include "legion.h"
#include "op_task_signature.h"
#include "profiling.h"
#include "runtime/config.h"
#include "serialization.h"
#include "tasks.h"
#include "utils/bidict.h"
#include "utils/optional.h"
#include "utils/stack_map.h"
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

/**
 * \class IsTrainable
 * \brief Denotes trainability
 * 
 * An enum class denoting the two types of argument references:
 *  (1) YES,
 *  (2) NO,
 * Used for InputVariadicParallelTensorDesc and bound using SimTaskBinding
*/
enum class IsTrainable { YES, NO };

/**
 * \class OpTensorSpec
 * \brief 
 * 
 * Deleted default constructor; Must pass in a TensorRole and integer to make an object; 
 * Has index and role (TensorRole)
*/
struct OpTensorSpec : public use_visitable_cmp<OpTensorSpec> {
public:
  OpTensorSpec() = delete;
  OpTensorSpec(TensorRole role, int idx);

public:
  TensorRole role;
  int idx;
};

/**
 * \fn OpTensorSpec input_tensor(int)
 * \param idx int denoting the index 
*/
OpTensorSpec input_tensor(int idx);

/**
 * \fn OpTensorSpec output_tensor(int idx)
 * \param idx int denoting the index
*/
OpTensorSpec output_tensor(int idx);

/**
 * \fn OpTensorSpec weight_tensor(int idx)
 * \param idx int denoting the index
*/
OpTensorSpec weight_tensor(int idx);

/**
 * \class OpArgRefType
 * \brief Enum class for operation argument reference type
 * 
 * An enum class denoting the types of operation argument references:
 *  (1) PER_DEVICE_OP_STATE,
 * Used for OpArgRef and resolve operation (parallel_computation_graph.cc)
*/
enum class OpArgRefType { PER_DEVICE_OP_STATE };

/**
 * \class OpArgRef
 * \brief Has reference type
 * 
 * Deleted default constructor—must pass in ref_type in order to create an object of ArgRef<T>
*/
template <typename T>
struct OpArgRef : public use_visitable_cmp<OpArgRef<T>> {
public:
  OpArgRef() = delete;
  OpArgRef(OpArgRefType ref_type) : ref_type(ref_type) {}

public:
  OpArgRefType ref_type;
};

/**
 * \fn OpArgRef<T> per_device_op_state()
 * \brief Returns OpArgRef with OpArgRefType
*/
template <typename T>
OpArgRef<T> per_device_op_state() {
  return OpArgRef<T>(OpArgRefType::PER_DEVICE_OP_STATE);
}

/**
 * \class OpArgRefSpec
 * \brief Struct for Operator Argument Reference Specification
 * 
 * Deleted default constructor—requires both params in order to create an object of OpArgRefSpec<T>. 
*/
struct OpArgRefSpec {
public:
  OpArgRefSpec() = delete;

  /**
   * \fn bool holds() const
   * \brief Checks for expected type
   * 
   * Returns TRUE if typeid(T) is the expected type (this->type); 
   * Used to check for correct/expected type and prevent possible errors;
  */
  template <typename T>
  bool holds() const {
    return std::type_index(typeid(T)) == this->type;
  }

  /**
   * \fn OpArgRefType const &get_ref_type() const
   * \brief Returns ref_type
  */
  OpArgRefType const &get_ref_type() const {
    return this->ref_type;
  }

  /**
   * \fn static OpArgRefSpec create(OpArgRef<T> const &r)
   * \brief Create OpArgRefSpec<T> from OpArgRef<T>
   * \param r OpArgRef used to get ref_type
   * 
   * Asserts serializability then returns object of OpArgRefSpec
  */
  template <typename T>
  static OpArgRefSpec create(OpArgRef<T> const &r) {
    static_assert(is_serializable<T>::value, "Type must be serializable");

    return OpArgRefSpec(std::type_index(typeid(T)), r.ref_type);
  }

private:
  OpArgRefSpec(std::type_index, OpArgRefType);

  std::type_index type;
  OpArgRefType ref_type;
};

using OpArgSpec = variant<ConcreteArgSpec,
                          IndexArgSpec,
                          OpArgRefSpec,
                          CheckedTypedFuture,
                          CheckedTypedFutureMap,
                          ArgRefSpec,
                          TaskInvocationSpec>;

/**
 * \class OpTaskBinding
 * \brief describes binding methods that insert argument specification
 * 
 * Has binding methods that insert argument specifications; 
 * Has properties: arg_bindings (std::unordered_map<...>)and tensor_bindings
 *  (std::unordered_map<...>);
*/
struct OpTaskBinding {
  OpTaskBinding() = default;

  static_assert(is_subeq_variant<IndexTaskArgSpec, OpArgSpec>::value, "");


  void bind(slot_id name, OpTensorSpec const &);
  void bind_grad(slot_id, OpTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    this->insert_arg_spec(name, ConcreteArgSpec::create(t));
  }

  template <typename T>
  void bind_arg(slot_id name, OpArgRef<T> const &ref) {
    this->insert_arg_spec(name, OpArgRefSpec::create(ref));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &f) {
    this->insert_arg_spec(name, CheckedTypedFuture::create(f));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &fm) {
    this->insert_arg_spec(name, CheckedTypedFutureMap::create(fm));
  }

  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> const &
      get_tensor_bindings() const;
  std::unordered_map<slot_id, OpArgSpec> const &get_arg_bindings() const;

private:
  void insert_arg_spec(slot_id name, OpArgSpec const &arg_spec) {
    assert(!contains_key(this->arg_bindings, name));
    arg_bindings.insert({name, arg_spec});
  }

  // template <typename T>
  // ArgSpec generate_arg_spec(T const &t) {
  //   static_assert(is_serializable<T>, "Type must be serializable");

  //   size_t pre_size = serializer.get_used_bytes();
  //   ff_task_serialize(serializer, t);
  //   size_t post_size = serializer.get_used_bytes();
  //   return {
  //     typeid(T),
  //     pre_size,
  //     post_size - pre_size
  //   };
  // }

  /* Legion::Serializer serializer; */
  std::unordered_map<slot_id, OpArgSpec> arg_bindings;
  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> tensor_bindings;
};

struct OpTaskInvocation : public use_visitable_cmp<OpTaskInvocation> {
public:
  OpTaskInvocation() = delete;
  OpTaskInvocation(task_id_t const &task_id, OpTaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  OpTaskBinding binding;
};

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

/* std::unordered_map<int, OpTensorSpec> get_regions_idxs(TaskArgumentFormat
 * const &); */

/* TaskArgumentFormat compile_task_invocation(OpTaskSignature const &,
 * OpTaskBinding const &); */

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::OpTensorSpec, role, idx);
VISITABLE_STRUCT(::FlexFlow::OpTensorSlotSpec, name, slot_type, tensor_role);

#endif

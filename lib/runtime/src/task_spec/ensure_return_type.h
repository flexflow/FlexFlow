#ifndef _FLEXFLOW_RUNTIME_SRC_ENSURE_RETURN_TYPE_H
#define _FLEXFLOW_RUNTIME_SRC_ENSURE_RETURN_TYPE_H

namespace FlexFlow {

/**
 * \fn RetType internal_ensure_return_type(InvocType const &invocation)
 * \brief ensure that return type has a value, if not throw errors
 * \param invocation InvocType used to get the task_id
 * 
 * Ensures that signature_return_type:
 * If so: return RetType(invocation)
 * If not: throw errors (e.g. has no return type, does not have asserted return type)
*/
template <typename InvocType, typename RetType, typename T>
RetType internal_ensure_return_type(InvocType const &invocation) {
  optional<std::type_index> signature_return_type = get_signature(invocation.task_id).get_return_type();
  std::type_index asserted_return_type = type_index<T>();
  if (!signature_return_type.has_value()) {
    throw mk_runtime_error("Task {} has no return type (asserted type {})",
                           asserted_return_type);
  }
  if (signature_return_type.value() != asserted_return_type) {
    throw mk_runtime_error("Task {} does not have asserted return type "
                           "(asserted type {}, signature type {})",
                           get_name(invocation.task_id),
                           asserted_return_type,
                           signature_return_type.value());
  }

  return RetType(invocation);
}

/**
 * \fn TypedTaskInvocation<T> ensure_return_type(TaskInvocation const &invocation)
 * \brief Preceding function to internal_ensure_return_type
 * \param invocation TaskInvocation used to call internal_ensure_return_type
 * 
 * Given TaskInvocation, call internal_ensure_return_type
*/
template <typename T>
TypedTaskInvocation<T> ensure_return_type(TaskInvocation const &invocation) {
  return internal_ensure_return_type<TaskInvocation, TypedTaskInvocation<T>, T>(
      invocation);
}

/**
 * \fn TypedIndexTaskInvocation<T> ensure_return_type(IndexTaskInvocation const &invocation)
 * \brief Preceding function to internal_ensure_return_type
 * \param invocation IndexTaskInvocation used to call internal_ensure_return_type
 * 
 * Given IndexTaskInvocation, call internal_ensure_return_type
*/
template <typename T>
TypedIndexTaskInvocation<T> ensure_return_type(IndexTaskInvocation const &invocation) {
  return internal_ensure_return_type<IndexTaskInvocation,
                                     TypedIndexTaskInvocation<T>,
                                     T>(invocation);
}

}

#endif

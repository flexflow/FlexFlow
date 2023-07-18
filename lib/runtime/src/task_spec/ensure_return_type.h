#ifndef _FLEXFLOW_RUNTIME_SRC_ENSURE_RETURN_TYPE_H
#define _FLEXFLOW_RUNTIME_SRC_ENSURE_RETURN_TYPE_H

namespace FlexFlow {

template <typename InvocType, typename RetType, typename T>
RetType internal_ensure_return_type(InvocType const &invocation) {
  optional<std::type_index> signature_return_type =
      get_signature(invocation.task_id).get_return_type();
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

template <typename T>
TypedStandardTaskInvocation<T>
    ensure_return_type(TaskInvocation const &invocation) {
  return internal_ensure_return_type<TaskInvocation,
                                     TypedStandardTaskInvocation<T>,
                                     T>(invocation);
}

template <typename T>
TypedIndexTaskInvocation<T>
    ensure_return_type(IndexTaskInvocation const &invocation) {
  return internal_ensure_return_type<IndexTaskInvocation,
                                     TypedIndexTaskInvocation<T>,
                                     T>(invocation);
}

} // namespace FlexFlow

#endif

#include "flexflow/runtime.h"
#include "runtime/model_training_instance.h"
#include "utils/expected.h"
#include "utils/ffi/opaque.h"

using namespace FlexFlow;

using Runtime = LibraryUtils<flexflow_runtime_error_t,
                             FLEXFLOW_RUNTIME_STATUS_OK,
                             FLEXFLOW_RUNTIME_ERROR_UNEXPECTED_EMPTY_HANDLE,
                             FLEXFLOW_RUNTIME_ERROR_DYNAMIC_ALLOCATION_FAILED>;
using R = Runtime;

template <typename T>
using err = R::err<T>;

#define CHECK_NONNULL(ptr)                                                     \
  if (ptr == nullptr) {                                                        \
    return FLEXFLOW_RUNTIME_ERROR_DYNAMIC_ALLOCATION_FAILED;                   \
  }                                                                            \
  static_assert(true, "");

static ModelTrainingInstance *
    unwrap_opaque(flexflow_model_training_instance_t opaque) {
  return opaque.ptr;
}

static TypedFuture<void> *unwrap_opaque(flexflow_void_future_t opaque) {
  return opaque.ptr;
}

char *flexflow_runtime_get_error_string(flexflow_runtime_error_t e) {
  char const *msg = nullptr;
  switch (e) {
    case FLEXFLOW_RUNTIME_STATUS_OK:
      break;
    case FLEXFLOW_RUNTIME_ERROR_UNKNOWN:
      msg = "Unknown error";
      break;
    case FLEXFLOW_RUNTIME_ERROR_UNEXPECTED_EMPTY_HANDLE:
      msg = "Expected non-empty handle but received empty handle (ptr == NULL)";
      break;
    default:
      msg = fmt("Unhandled case for value {}. Please report to the FlexFlow "
                "developers.",
                e);
  }
  return const_cast<char *>(msg);
}

flexflow_runtime_error_t flexflow_model_training_instance_forward(
    flexflow_model_training_instance_t opaque, flexflow_void_future_t *out) {
  ModelTrainingInstance *training_instance = unwrap_opaque(opaque);
  CHECK_NONNULL(training_instance);

  return R::output_stored(forward(*training_instance); out);
}

flexflow_runtime_error_t
    flexflow_void_future_destroy(flexflow_void_future_t opaque) {
  return R::deallocate_opaque(opaque);
}

flexflow_runtime_error_t
    flexflow_wait_on_void_future(flexflow_void_future_t opaque) {
  TypedFuture<void> *void_future = unwrap_opaque(opaque);
  CHECK_NONNULL(void_future);

  void_future.get_result<void>();
  return FLEXFLOW_RUNTIME_STATUS_OK;
}

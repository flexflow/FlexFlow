#ifndef _FLEXFLOW_RUNTIME_SRC_DEVICE_SPECIFIC_ARG_H
#define _FLEXFLOW_RUNTIME_SRC_DEVICE_SPECIFIC_ARG_H

#include "serialization.h"
#include "task_argument_accessor.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename T>
struct DeviceSpecificArg {

  DeviceSpecificArg(T *) {
    NOT_IMPLEMENTED();
  }

  T *get(TaskArgumentAccessor const &accessor) const {
    if (accessor.get_device_idx() != this->device_idx) {
      throw mk_runtime_error("Invalid access to DeviceSpecificArg: attempted "
                             "device_idx {} != correct device_idx {})",
                             accessor.get_device_idx(),
                             this->device_idx);
    }
  }

private:
  T *ptr;
  size_t device_idx;
};

// manually force serialization to make DeviceSpecificArgs trivially
// serializable
template <typename T>
struct is_trivially_serializable<DeviceSpecificArg<T>> : std::true_type {};

} // namespace FlexFlow

#endif

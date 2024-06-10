#ifndef _FLEXFLOW_LOCAL_EXECUTION_DEVICE_SPECIFIC_H
#define _FLEXFLOW_LOCAL_EXECUTION_DEVICE_SPECIFIC_H

#include "local-execution/serialization.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename T>
struct DeviceSpecific {

  DeviceSpecific() = delete;

  template <typename... Args>
  static DeviceSpecific<T> create(Args &&...args) {
    NOT_IMPLEMENTED();
  }

  T const *get(size_t curr_device_idx) const {
    if (curr_device_idx != this->device_idx) {
      throw mk_runtime_error("Invalid access to DeviceSpecific: attempted "
                             "device_idx {} != correct device_idx {})",
                             curr_device_idx,
                             this->device_idx);
    }
    return this->ptr;
  }

  // TODO: can modify ptr

private:
  DeviceSpecific(T *ptr, size_t device_idx)
      : ptr(ptr), device_idx(device_idx) {}

  T *ptr;
  size_t device_idx;
};

// manually force serialization to make DeviceSpecific trivially
// serializable
// template <typename T>
// struct is_trivially_serializable<DeviceSpecific<T>> : std::true_type {};

} // namespace FlexFlow

#endif

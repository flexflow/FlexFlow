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
    size_t device_idx = 0;
    return DeviceSpecific<T>(std::make_shared<T>(std::forward<Args>(args)...),
                             device_idx);
  }

  bool operator==(DeviceSpecific const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(DeviceSpecific const &other) const {
    return this->tie() != other.tie();
  }

  T const *get(size_t curr_device_idx) const {
    if (curr_device_idx != this->device_idx) {
      throw mk_runtime_error("Invalid access to DeviceSpecific: attempted "
                             "device_idx {} != correct device_idx {})",
                             curr_device_idx,
                             this->device_idx);
    }
    return (T const *)this->ptr.get();
  }

  // TODO: can modify ptr

private:
  DeviceSpecific(std::shared_ptr<T> ptr, size_t device_idx)
      : ptr(ptr), device_idx(device_idx) {}

  std::shared_ptr<T> ptr;
  size_t device_idx;

  std::tuple<decltype(ptr) const &, decltype(device_idx) const &> tie() const {
    return std::tie(this->ptr, this->device_idx);
  }
};

// manually force serialization to make DeviceSpecific trivially
// serializable
// template <typename T>
// struct is_trivially_serializable<DeviceSpecific<T>> : std::true_type {};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LOCAL_EXECUTION_DEVICE_SPECIFIC_H
#define _FLEXFLOW_LOCAL_EXECUTION_DEVICE_SPECIFIC_H

#include "task-spec/global_device_id_t.dtg.h"
#include "utils/hash/tuple.h"
#include "utils/not_implemented.h"
#include <libassert/assert.hpp>
#include <nlohmann/json.hpp>

namespace FlexFlow {

template <typename T>
struct DeviceSpecific {
  DeviceSpecific() = delete;

  template <typename... Args>
  static DeviceSpecific<T> create(global_device_id_t const &device_idx,
                                  Args &&...args) {
    return DeviceSpecific<T>(std::make_shared<T>(std::forward<Args>(args)...),
                             device_idx);
  }

  bool operator==(DeviceSpecific const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(DeviceSpecific const &other) const {
    return this->tie() != other.tie();
  }

  bool operator<(DeviceSpecific const &other) const {
    return this->tie() < other.tie();
  }

  bool operator<=(DeviceSpecific const &other) const {
    return this->tie() <= other.tie();
  }

  bool operator>(DeviceSpecific const &other) const {
    return this->tie() > other.tie();
  }

  bool operator>=(DeviceSpecific const &other) const {
    return this->tie() >= other.tie();
  }

  T const *get(global_device_id_t const &curr_device_idx) const {
    ASSERT(curr_device_idx == this->device_idx);
    return (T const *)this->ptr.get();
  }

private:
  DeviceSpecific(std::shared_ptr<T> ptr, global_device_id_t const &device_idx)
      : ptr(ptr), device_idx(device_idx) {}

private:
  std::shared_ptr<T> ptr;
  global_device_id_t device_idx;

private:
  std::tuple<decltype(ptr) const &, decltype(device_idx) const &> tie() const {
    return std::tie(this->ptr, this->device_idx);
  }

  friend struct ::std::hash<DeviceSpecific<T>>;

  friend std::string format_as(DeviceSpecific<T> const &d) {
    return fmt::format("DeviceSpecific({:p}, {})",
                       static_cast<void *>(d.ptr.get()),
                       d.device_idx);
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &s, DeviceSpecific<T> const &d) {
  return (s << fmt::to_string(d));
}

template <typename T>
void to_json(nlohmann::json &j, DeviceSpecific<T> const &d) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::DeviceSpecific<T>> {
  size_t operator()(::FlexFlow::DeviceSpecific<T> const &x) const {
    return get_std_hash(x.tie());
  }
};

} // namespace std

#endif

#ifndef _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_VALUE_PTR_T_H
#define _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_VALUE_PTR_T_H

#include "is_clonable.h"
#include <utility>

namespace FlexFlow {

template <typename T>
struct value_ptr {
  static_assert(is_clonable_v<T>,
                "value_ptr requires the type to have a clone() method");

  value_ptr(T *ptr) : ptr(ptr) {}
  value_ptr(value_ptr<T> const &other) : ptr(other.ptr->clone()) {}

  T *get() const {
    return ptr;
  }

  T *operator->() const {
    return ptr;
  }

  T &operator*() const {
    return *ptr;
  }

  friend void swap(value_ptr const &lhs, value_ptr const &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

private:
  T *ptr;
};

template <typename T, typename... Args>
std::enable_if_t<std::is_constructible_v<T, Args...>, value_ptr<T>>
    make_value_ptr(Args &&...args) {
  return {new T(std::forward<Args>(args)...)};
}

template <typename T, typename... Args>
std::enable_if_t<!std::is_constructible_v<T, Args...>, value_ptr<T>>
    make_value_ptr(Args &&...args) {
  return {new T{std::forward<Args>(args)...}};
}

} // namespace FlexFlow

#endif

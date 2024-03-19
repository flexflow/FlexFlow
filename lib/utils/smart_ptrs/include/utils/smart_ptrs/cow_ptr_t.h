#ifndef _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_COW_PTR_T_H
#define _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_COW_PTR_T_H

#include "is_clonable.h"
#include <memory>
#include <type_traits>

namespace FlexFlow {

template <typename T>
struct cow_ptr_t {
  static_assert(is_clonable_v<T>,
                "cow_ptr_t requires the type to have a clone() method");

  cow_ptr_t(std::shared_ptr<T> const &ptr) : ptr(ptr) {}
  cow_ptr_t(std::shared_ptr<T> &&ptr) : ptr(std::move(ptr)) {}
  cow_ptr_t(T const &val) : ptr(std::make_shared<T>(val)) {}
  cow_ptr_t(cow_ptr_t const &other) {
    this->ptr = other.ptr;
  }

  cow_ptr_t &operator=(cow_ptr_t other) {
    swap(*this, other);
    return *this;
  }

  T const &operator*() const {
    return *this->get();
  }

  template <typename TT, typename = std::enable_if_t<std::is_base_of_v<TT, T>>>
  operator cow_ptr_t<TT>() const {
    return cow_ptr_t<TT>(this->ptr);
  }

  std::shared_ptr<T const> operator->() const {
    return this->get();
  }

  T const *get_raw_unsafe() const {
    return this->ptr.get();
  }

  std::shared_ptr<T const> get() const {
    return this->ptr;
  }

  std::shared_ptr<T> get_mutable() const {
    if (!this->has_unique_access()) {
      this->ptr = std::shared_ptr<T>(this->ptr->clone());
    }
    return this->ptr;
  }

  friend bool operator==(cow_ptr_t const &lhs, cow_ptr_t const &rhs) {
    return lhs.ptr == rhs.ptr;
  }

  friend bool operator!=(cow_ptr_t const &lhs, cow_ptr_t const &rhs) {
    return lhs.ptr != rhs.ptr;
  }

  friend void swap(cow_ptr_t &lhs, cow_ptr_t &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }
private:
  bool has_unique_access() const {
    return this->ptr.use_count() == 1;
  }

  mutable std::shared_ptr<T> ptr = nullptr;
};

template <typename T, typename... Args>
cow_ptr_t<T> make_cow_ptr(Args &&...args) {
  return {std::make_shared<T>(std::forward<Args>(args)...)};
}

template <class To, class From>
cow_ptr_t<To> static_pointer_cast(cow_ptr_t<From> const &x) {
  std::shared_ptr<From> inner = std::const_pointer_cast<From>(x.get());
  return cow_ptr_t{std::static_pointer_cast<To>(inner)};
}

template <class To, class From>
cow_ptr_t<To> dynamic_pointer_cast(cow_ptr_t<From> const &x) {
  std::shared_ptr<From> inner = std::const_pointer_cast<From>(x.get());
  return cow_ptr_t{std::dynamic_pointer_cast<To>(inner)};
}

template <class To, class From>
cow_ptr_t<To> reinterpret_pointer_cast(cow_ptr_t<From> const &x) {
  std::shared_ptr<From> inner = std::const_pointer_cast<From>(x.get());
  return cow_ptr_t{std::reinterpret_pointer_cast<To>(inner)};
}

} // namespace FlexFlow

#endif

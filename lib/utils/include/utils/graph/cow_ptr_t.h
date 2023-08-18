#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_COW_PTR_T_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_COW_PTR_T_H

#include "utils/type_traits.h"
#include "utils/unique.h"
#include "utils/variant.h"
#include <memory>

namespace FlexFlow {

template <typename T>
struct cow_ptr_t {
  // static_assert(is_clonable<T>::value,
  //               "cow_ptr_t requires the type to have a clone() method"); //
  //               TODO:
  //               https://github.com/flexflow/FlexFlow/issues/909#issue-1833470024

  cow_ptr_t(std::shared_ptr<T> ptr) : ptr(std::move(ptr)) {}
  cow_ptr_t(std::unique_ptr<T> ptr) : ptr(std::move(ptr)) {}
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

  std::shared_ptr<T const> operator->() const {
    return this->get();
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

} // namespace FlexFlow

#endif

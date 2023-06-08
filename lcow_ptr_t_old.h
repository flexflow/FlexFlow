#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_COW_PTR_T_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_COW_PTR_T_H

#include <memory>
#include "utils/variant.h"
#include "utils/unique.h"
#include "utils/type_traits.h"

namespace FlexFlow {

template <typename T>
struct cow_ptr_t {
  cow_ptr_t() {
    this->set_unique(nullptr);
  }
  cow_ptr_t(std::shared_ptr<T> ptr) {
    this->set_shared(std::move(ptr));
  }
  cow_ptr_t(std::unique_ptr<T> ptr) {
    this->set_unique(std::move(ptr));
  }
  cow_ptr_t(T const &val) {
    this->set_unique(make_unique<T>(val));
  }
  cow_ptr_t(cow_ptr_t const &other) {
    this->set_shared(other.get_shared_ptr());  
  }
  cow_ptr_t &operator=(cow_ptr_t other) {
    swap(*this, other); 
    return *this;
  }

  using shared_t = std::shared_ptr<T const>;
  using unique_t = std::unique_ptr<T>;

  T const *get() const {
    return &this->ref();
  }

  T const &ref() const {
    if (this->has_unique_access()) {
      return *this->get_unique();
    } else {
      return *this->get_shared();
    }
  }

  T const &operator*() const {
    return this->get();
  }

  T const *operator->() const {
    return this->get();
  }

std::shared_ptr<T const> get_shared_ptr() const {
  if (this->has_unique_access()) {
    auto clone_unique = [](const unique_t& ptr) -> unique_t {
      return make_unique<T>(*ptr);
    };

    unique_t uniquePtr = clone_unique(this->get_unique());
    this->set_shared(std::make_shared<T const>(std::move(uniquePtr)));
  }
  return this->get_shared();
}

T* mutable_ptr() const {
  if (this->has_unique_access()) {
    return this->get_unique().get();
  } else {
    this->set_unique(unique_t(this->get_shared()->clone()));
    return this->get_unique().get();
  }
}

T& mutable_ref() const {
  return *mutable_ptr();
}

  bool has_unique_access() const { 
    return holds_alternative<unique_t>(this->ptr);
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

void set_shared(shared_t ptr) const {
    this->ptr = variant<shared_t>(std::move(ptr));
}


  void set_unique(unique_t ptr) {
    this->ptr =  variant<unique_t>(std::move(ptr));
    // this->ptr = variant<std::shared_ptr<T const>>(std::move(ptr));
    //this->ptr = variant<unique_t>(std::move(ptr));
  }

std::unique_ptr<T> const& get_unique() const {
  return std::get<std::unique_ptr<T>>(this->ptr);
}

std::shared_ptr<T const> const& get_shared() const {
  return std::get<std::shared_ptr<T const>>(this->ptr);
}
  mutable variant<
    std::unique_ptr<T>,
    std::shared_ptr<T const>
  > ptr;
};

}

#endif

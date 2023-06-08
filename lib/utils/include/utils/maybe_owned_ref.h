#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_MAYBE_OWNED_REF_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_MAYBE_OWNED_REF_H

#include "utils/variant.h"
#include <memory>

namespace FlexFlow {

template <typename T> struct maybe_owned_ref {
  maybe_owned_ref() = delete;
  maybe_owned_ref(T *);
  maybe_owned_ref(std::shared_ptr<T>);

  T &get() const {
    if (holds_alternative<T *>(this->_ptr)) {
      return *mpark::get<T *>(this->_ptr);
    } else {
      return *mpark::get<std::shared_ptr<T>>(this->_ptr);
    }
  }

  operator T &() const { return this->get(); }

private:
  variant<T *, std::shared_ptr<T>> _ptr;
};

static_assert(is_copy_constructible<maybe_owned_ref<int>>::value,
              "maybe_owned_ref must be copy constructible");

} // namespace FlexFlow

#endif

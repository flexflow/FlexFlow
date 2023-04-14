#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_MAYBE_OWNED_REF_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_MAYBE_OWNED_REF_H

#include "utils/variant.h"
#include <memory>

namespace FlexFlow {

template <typename T>
struct maybe_owned_ref {
  maybe_owned_ref() = delete;
  maybe_owned_ref(T*);
  maybe_owned_ref(std::shared_ptr<T>);
  maybe_owned_ref(std::unique_ptr<T>);

  operator T &() const {
    if (holds_alternative<T*>(this->_ptr)) {
      return *get<T*>(this->_ptr);
    } else if (holds_alternative<std::shared_ptr<T>>) {
      return *get<std::shared_ptr<T>>(this->_ptr);
    } else {
      return *get<std::unique_ptr<T>>(this->_ptr);
    }
  }
private:
  variant<T *, std::shared_ptr<T>, std::unique_ptr<T>> _ptr;
};

}

#endif

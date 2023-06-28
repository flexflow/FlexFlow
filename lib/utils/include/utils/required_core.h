#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H

namespace FlexFlow {

template <typename T>
struct required {
public:
  required() = delete;
  required(T const &t) : value(t) { };
  required(T &&t) : value(t) { };

  using value_type = T;

  operator T const &() const { 
    return this->value;
  }

public:
  T value;
};

template <typename T>
using req = required<T>;

template <typename T> 
struct remove_req {
  using type = T;
};

template <typename T>
struct remove_req<req<T>> {
  using type = T;
};

template <typename T>
using remove_req_t = typename remove_req<T>::type;

}

#endif

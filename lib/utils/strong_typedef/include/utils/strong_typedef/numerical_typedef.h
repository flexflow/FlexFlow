#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_NUMERICAL_TYPEDEF_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_NUMERICAL_TYPEDEF_H

#include "strong_typedef.h"

namespace FlexFlow {

template <typename StrongTypedef, typename T>
struct numerical_typedef : strong_typedef<StrongTypedef, T> {
  using strong_typedef<StrongTypedef, T>::strong_typedef;

  friend StrongTypedef &operator+=(StrongTypedef &lhs, T const &rhs) {
    static_cast<T &>(lhs) += static_cast<T const &>(rhs);
    return lhs;
  }

  friend StrongTypedef &operator++(StrongTypedef &lhs) {
    static_cast<T &>(lhs) += static_cast<T>(1);
    return lhs;
  }

  friend StrongTypedef operator++(StrongTypedef &lhs, int) {
    StrongTypedef tmp = lhs;
    ++lhs;
    return tmp;
  }

  friend StrongTypedef operator+(StrongTypedef const &lhs, T const &rhs) {
    return StrongTypedef(lhs.value() + rhs);
  }

  friend StrongTypedef operator+(T const &lhs, StrongTypedef const &rhs) {
    return (rhs + lhs);
  }

  friend StrongTypedef operator-=(StrongTypedef &lhs, T const &rhs) {
    static_cast<T &>(lhs) -= static_cast<T const &>(rhs);
  }

  friend StrongTypedef &operator--(StrongTypedef &lhs) {
    static_cast<T &>(lhs) -= static_cast<T>(1);
    return lhs;
  }

  friend StrongTypedef operator--(StrongTypedef &lhs, int) {
    StrongTypedef tmp = lhs;
    --lhs;
    return tmp;
  }

  friend StrongTypedef operator-(StrongTypedef const &lhs, T const &rhs) {
    return StrongTypedef(lhs.value() + rhs);
  }

  friend bool operator<(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() < rhs;
  }

  friend bool operator==(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() == rhs;
  }

  friend bool operator>(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() > rhs;
  }

  friend bool operator>=(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() >= rhs;
  }

  friend bool operator!=(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() != rhs;
  }

  friend bool operator<=(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() <= rhs;
  }

  friend bool operator<(T const &lhs, StrongTypedef const &rhs) {
    return lhs < rhs.value();
  }

  friend bool operator==(T const &lhs, StrongTypedef const &rhs) {
    return lhs == rhs.value();
  }

  friend bool operator>(T const &lhs, StrongTypedef const &rhs) {
    return lhs > rhs.value();
  }

  friend bool operator<=(T const &lhs, StrongTypedef const &rhs) {
    return lhs <= rhs.value();
  }

  friend bool operator!=(T const &lhs, StrongTypedef const &rhs) {
    return lhs != rhs.value();
  }

  friend bool operator>=(T const &lhs, StrongTypedef const &rhs) {
    return lhs >= rhs.value();
  }
};

} // namespace FlexFlow

#endif

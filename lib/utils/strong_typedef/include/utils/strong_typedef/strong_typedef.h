#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_STRONG_TYPEDEF_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_STRONG_TYPEDEF_H

#include <functional>
#include <string>
#include <type_traits>
#include "utils/type_traits_extra/is_static_castable.h"

namespace FlexFlow {

// derived from https://www.foonathan.net/2016/10/strong-typedefs/
template <typename Tag, typename T>
class strong_typedef {
public:
  strong_typedef() = delete;

  explicit strong_typedef(T const &value) : value_(value) {}

  explicit strong_typedef(T &&value) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : value_(std::move(value)) {}

  explicit operator T &() noexcept {
    return value_;
  }

  explicit operator T const &() const noexcept {
    return value_;
  }

  template <typename TT,
            typename std::enable_if<(is_static_castable<T, TT>::value &&
                                     !std::is_same<T, TT>::value),
                                    bool>::type = true>
  explicit operator TT() const {
    return static_cast<TT>(this->value_);
  }

  template <typename TT,
            typename std::enable_if<(std::is_convertible<T, TT>::value &&
                                     !std::is_same<T, TT>::value),
                                    bool>::type = true>
  operator TT() const {
    return (this->value_);
  }

  friend void swap(strong_typedef &a, strong_typedef &b) noexcept {
    using std::swap;
    swap(static_cast<T &>(a), static_cast<T &>(b));
  }

  friend bool operator==(strong_typedef const &lhs, strong_typedef const &rhs) {
    return lhs.value() == rhs.value();
  }

  friend bool operator!=(strong_typedef const &lhs, strong_typedef const &rhs) {
    return lhs.value() != rhs.value();
  }

  friend bool operator<(strong_typedef const &lhs, strong_typedef const &rhs) {
    return lhs.value() < rhs.value();
  }

  T const &value() const noexcept {
    return value_;
  }

  template <typename F>
  strong_typedef fmap(F const &f) {
    static_assert(
        std::is_same<decltype(std::declval<F>()(std::declval<T const &>())),
                     T>::value,
        "Function must return an value of the underlying type");

    return strong_typedef(f(this->value_));
  }

private:
  T value_;
};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H

#include "utils/backports/type_identity.h"
#include "utils/type_traits_extra/is_equal_comparable.h"
#include "utils/type_traits_extra/is_neq_comparable.h"
#include "utils/type_traits_extra/is_static_castable.h"
#include <type_traits>
#include <vector>

namespace FlexFlow {

template <typename T>
struct required_wrapper_impl {
public:
  required_wrapper_impl() = delete;
  required_wrapper_impl(T const &t) : m_value(t) {}
  required_wrapper_impl(T &&t) : m_value(t) {}

  template <typename TT>
  required_wrapper_impl(TT const &tt,
                        std::enable_if_t<std::is_convertible_v<TT, T>> * = 0)
      : m_value(static_cast<T>(tt)) {}

  using value_type = T;

  operator T const &() const {
    return this->m_value;
  }

  template <typename TT,
            typename std::enable_if_t<(is_static_castable_v<T, TT> &&
                                       !std::is_same_v<T, TT>),
                                      bool> = true>
  explicit operator TT() const {
    return static_cast<TT>(this->m_value);
  }

private:
  T m_value;
};

template <typename T>
T format_as(required_wrapper_impl<T> const &r) {
  return static_cast<T>(r);
}

template <typename T>
struct required_inheritance_impl : public T {
  static_assert(std::is_class_v<T>);

  using T::T;
  required_inheritance_impl() = delete;
  required_inheritance_impl(T const &t) : T(t) {}
  required_inheritance_impl(T &&t) : T(t) {}

  required_inheritance_impl(required_inheritance_impl<T> const &) = default;
  required_inheritance_impl(required_inheritance_impl<T> &&) = default;

  required_inheritance_impl<T> &
      operator=(required_inheritance_impl<T> const &) = default;
  required_inheritance_impl<T> &
      operator=(required_inheritance_impl<T> &&) = default;

  template <typename TT>
  required_inheritance_impl(TT const &tt,
                            std::enable_if_t<std::is_convertible_v<TT, T> &&
                                             !std::is_same_v<TT, T>> * = 0)
      : required_inheritance_impl(static_cast<T>(tt)) {}

  template <typename TTT,
            std::enable_if_t<(is_static_castable_v<T, TTT> &&
                              !std::is_same_v<T, TTT>),
                             bool> = true>
  explicit operator TTT() const {
    return static_cast<TTT>(static_cast<T>(*this));
  }
};

template <typename T, typename = std::enable_if_t<is_equal_comparable_v<T>>>
bool operator==(required_inheritance_impl<T> const &lhs,
                required_inheritance_impl<T> const &rhs) {
  return static_cast<T>(lhs) == static_cast<T>(rhs);
}

template <typename T, typename = std::enable_if_t<is_neq_comparable_v<T>>>
bool operator!=(required_inheritance_impl<T> const &lhs,
                required_inheritance_impl<T> const &rhs) {
  return static_cast<T>(lhs) != static_cast<T>(rhs);
}

template <typename T>
T format_as(required_inheritance_impl<T> const &r) {
  return static_cast<T>(r);
}

template <typename T, typename Enable = void>
struct required : public required_wrapper_impl<T> {
  static_assert(std::is_default_constructible_v<T>);

  using required_wrapper_impl<T>::required_wrapper_impl;
};

template <typename T>
T format_as(required<T> const &r) {
  return static_cast<T>(r);
}

template <typename T>
struct required<T, std::enable_if_t<std::is_class_v<T>>>
    : public required_inheritance_impl<T> {
  static_assert(std::is_default_constructible_v<T>);

  using required_inheritance_impl<T>::required_inheritance_impl;
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

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::req<T>> {
  size_t operator()(::FlexFlow::req<T> const &r) const {
    return std::hash<T>{}(static_cast<T>(r));
  }
};

} // namespace std

#endif

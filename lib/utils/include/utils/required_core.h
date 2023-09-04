#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H

#include "fmt.decl.h"
#include "hash-utils-core.h"
#include "test_types.h"
#include "type_traits_core.h"
#include <vector>

namespace FlexFlow {

template <typename T, typename Enable = void, typename...>
struct enable_if_valid {};

template <typename T, typename... Args>
struct enable_if_valid<T, void_t<Args...>, Args...> : type_identity<T> {};

template <typename T>
struct required_wrapper_impl {
public:
  required_wrapper_impl() = delete;
  required_wrapper_impl(T const &t) : m_value(t) {}
  required_wrapper_impl(T &&t) : m_value(t) {}

  template <typename TT>
  required_wrapper_impl(
      TT const &tt,
      typename std::enable_if<std::is_convertible<TT, T>::value>::type * = 0)
      : m_value(static_cast<T>(tt)) {}

  using value_type = T;

  operator T const &() const {
    return this->m_value;
  }

  template <typename TT,
            typename std::enable_if<(is_static_castable<T, TT>::value &&
                                     !std::is_same<T, TT>::value),
                                    bool>::type = true>
  explicit operator TT() const {
    return static_cast<TT>(this->m_value);
  }

private:
  T m_value;
};

template <typename T, typename = std::enable_if_t<is_fmtable_v<T>>>
T format_as(required_wrapper_impl<T> const &r) {
  return static_cast<T>(r);
}

template <typename T>
struct required_inheritance_impl : public T {
  static_assert(std::is_class<T>::value, "");

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
  required_inheritance_impl(
      TT const &tt,
      typename std::enable_if<std::is_convertible<TT, T>::value &&
                              !std::is_same<TT, T>::value>::type * = 0)
      : required_inheritance_impl(static_cast<T>(tt)) {}

  template <typename TTT,
            typename std::enable_if<(is_static_castable<T, TTT>::value &&
                                     !std::is_same<T, TTT>::value),
                                    bool>::type = true>
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

template <typename T, typename = std::enable_if_t<is_fmtable_v<T>>>
T format_as(required_inheritance_impl<T> const &r) {
  return static_cast<T>(r);
}

template <typename T, typename Enable = void>
struct required : public required_wrapper_impl<T> {
  static_assert(std::is_default_constructible_v<T>);

  using required_wrapper_impl<T>::required_wrapper_impl;
};

template <typename T, typename = std::enable_if_t<is_fmtable_v<T>>>
T format_as(required<T> const &r) {
  return static_cast<T>(r);
}

template <typename T>
struct required<T, typename std::enable_if<std::is_class<T>::value>::type>
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

static_assert(
    is_equal_comparable<required_inheritance_impl<std::vector<int>>>::value,
    "");
CHECK_FMTABLE(required_wrapper_impl<int>);
CHECK_FMTABLE(required_wrapper_impl<test_types::fmtable>);
CHECK_FMTABLE(required_inheritance_impl<std::string>);
CHECK_FMTABLE(required_inheritance_impl<test_types::fmtable>);
CHECK_FMTABLE(required<int>);
CHECK_FMTABLE(required<std::string>);

static_assert(std::is_copy_constructible<req<int>>::value, "");

static_assert(std::is_convertible<req<int>, int>::value, "");
static_assert(is_static_castable<req<void *>, int *>::value, "");

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::req<T>> {
  size_t operator()(::FlexFlow::req<T> const &r) const {
    return get_std_hash(static_cast<T>(r));
  }
};

} // namespace std

#endif

#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H

#include "fmt.decl.h"
#include "hash-utils.h"
#include "test_types.h"
#include "type_traits_core.h"
#include <vector>

namespace FlexFlow {

template <typename T, typename Enable = void, typename...>
struct enable_if_valid {};

template <typename T, typename... Args>
struct enable_if_valid<T, void_t<Args...>, Args...> : type_identity<T> {};

/* required_wrapper_impl<decltype(std::declval<T>() + std::declval<T>())>> */
/* operator+(required_wrapper_impl<T> const &lhs, required_wrapper_impl<T> const
 * &rhs) { */
/*   /1* return 1; *1/ */
/* } */

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

  friend T format_as(required_wrapper_impl<T> const &r) {
    return static_cast<T>(r);
  }

  /* T const &operator*() const { */
  /*   return this->m_value; */
  /* } */

  /* T const *operator->() const { */
  /*   return &this->m_value; */
  /* } */

  template <typename TT = T>
  enable_if_t<is_neq_comparable<TT>::value, bool>
      operator==(required_wrapper_impl const &rhs) const {
    return this->m_value == rhs.m_value;
  }

  template <typename TT = T>
  enable_if_t<is_neq_comparable<TT>::value, bool>
      operator==(TT const &rhs) const {
    return this->m_value == rhs;
  }

  /* friend enable_if_t<is_equal_comparable<T>::value, bool> */
  /*   operator==(required_wrapper_impl<T> const &lhs, T const &rhs) { */
  /*   return lhs.m_value == rhs; */
  /* } */

  /* friend enable_if_t<is_equal_comparable<T>::value, bool> */
  /*   operator==(T const &lhs, required_wrapper_impl<T> const &rhs) { */
  /*   return lhs == rhs.m_value; */
  /* } */

  template <typename TT = T>
  enable_if_t<is_neq_comparable<TT>::value, bool>
      operator!=(required_wrapper_impl const &rhs) const {
    return this->m_value != rhs.m_value;
  }

  /* friend enable_if_t<is_plusable<T>::value,
   * required_wrapper_impl<decltype(std::declval<T>() + std::declval<T>())>> */
  /* operator+(required_wrapper_impl<T> const &lhs, required_wrapper_impl<T>
   * const &rhs) { */
  /*   /1* return 1; *1/ */
  /* } */
  /* required_wrapper_impl<Out> */
  /* operator+(required_wrapper_impl const &rhs) { */
  /*   Out o = this->m_value + rhs.m_value; */
  /*   return required_wrapper_impl<Out>{o}; */
  /* } */

  /* template <typename TT = T, */
  /*           enable_if_t<is_minusable<TT>::value> = true> */
  /* required_wrapper_impl operator-(required_wrapper_impl const &rhs) { */
  /*   return {this->m_value - rhs.m_value}; */
  /* } */

  /* template <typename TT = T, */
  /*           enable_if_t<is_timesable<TT>::value> = true> */
  /* required_wrapper_impl operator*(required_wrapper_impl const &rhs) { */
  /*   return {this->m_value * rhs.m_value}; */
  /* } */

  /* bool operator<(T const &other) const { */
  /*   return this->m_value < other; */
  /* } */

  /* bool operator>(T const &other) const { */
  /*   return this->m_value > other; */
  /* } */

private:
  T m_value;
};

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

  friend enable_if_t<is_equal_comparable<T>::value, bool>
      operator==(required_inheritance_impl<T> const &lhs,
                 required_inheritance_impl<T> const &rhs) {
    return static_cast<T>(lhs) == static_cast<T>(rhs);
  }

  friend enable_if_t<is_neq_comparable<T>::value, bool>
      operator!=(required_inheritance_impl<T> const &lhs,
                 required_inheritance_impl<T> const &rhs) {
    return static_cast<T>(lhs) != static_cast<T>(rhs);
  }

  friend std::string format_as(required_inheritance_impl<T> const &r) {
    return "";
    /* static_assert(is_fmtable<T>::value, ""); */

    /* return static_cast<T>(r); */
  }

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

template <typename T, typename Enable = void>
struct required : public required_wrapper_impl<T> {
  using required_wrapper_impl<T>::required_wrapper_impl;
};

template <typename T>
struct required<T, typename std::enable_if<std::is_class<T>::value>::type>
    : public required_inheritance_impl<T> {
  using required_inheritance_impl<T>::required_inheritance_impl;
};

template <typename T>
using req = required<T>;

template <typename T>
struct delegate_ostream_operator<req<T>> : std::true_type {};

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
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(
    required_inheritance_impl<test_types::well_behaved_value_type>);
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(
    required_wrapper_impl<test_types::well_behaved_value_type>);

/* static_assert(std::is_same<decltype(std::declval<required_wrapper_impl<int>>()
 * + std::declval<required_wrapper_impl<int>>()),
 * required_wrapper_impl<int>>::value, ""); */

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

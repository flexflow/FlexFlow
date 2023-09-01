#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H

#include "hash-utils-core.h"
#include "type_traits_core.h"
#include <vector>

namespace FlexFlow {

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

  /* T const &operator*() const { */
  /*   return this->m_value; */
  /* } */

  /* T const *operator->() const { */
  /*   return &this->m_value; */
  /* } */

  /* bool operator==(T const &other) const { */
  /*   return this->m_value == other; */
  /* } */

  /* bool operator!=(T const &other) const { */
  /*   return this->m_value != other; */
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
  required_inheritance_impl(T const &);
  required_inheritance_impl(T &&t);

  template <typename TT>
  required_inheritance_impl(
      TT const &tt,
      typename std::enable_if<std::is_convertible<TT, T>::value &&
                              !std::is_same<TT, T>::value>::type * = 0)
      : required_inheritance_impl(static_cast<T>(tt)) {}

  operator T() const;

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
struct remove_req {
  using type = T;
};

template <typename T>
struct remove_req<req<T>> {
  using type = T;
};

template <typename T>
using remove_req_t = typename remove_req<T>::type;

static_assert(std::is_convertible<req<int>, int>::value, "");
static_assert(is_static_castable<req<void *>, int *>::value, "");
static_assert(
    std::is_same<
        void_t<decltype(std::declval<req<int>>() == std::declval<int>())>,
        void>::value,
    "");
static_assert(is_list_initializable<req<bool>, bool>::value, "");
static_assert(
    std::is_same<
        void_t<decltype(std::declval<req<int>>() + std::declval<int>())>,
        void>::value,
    "");

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

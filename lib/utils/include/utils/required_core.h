#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_CORE_H

#include "type_traits_core.h"
#include <type_traits>
#include <vector>

namespace FlexFlow {

template <typename T>
struct required {
public:
  required() = delete;
  required(T const &t) : m_value(t) {}
  required(T &&t) : m_value(t) {}

  template <typename TT>
  required(
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
  /*   return this-> */
  /* } */

  T const &value() const {
    return this->m_value;
  }

private:
  T m_value;
};


template <typename T, typename Enable = void> struct required_v2 : public required<T> { using required<T>::required; };

template <typename T>
struct required_v2<
  T, 
  typename std::enable_if<std::is_class<T>::value>::type
> : public T {

  using T::T;
  required_v2() = delete;
  required_v2(T const &);
  required_v2(T &&t);

  template <typename TT>
  required_v2(
      TT const &tt,
      typename std::enable_if<std::is_convertible<TT, T>::value>::type * = 0)
      : required_v2(static_cast<T>(tt)) { }

  operator T() const;

  template <typename TTT,
            typename std::enable_if<(is_static_castable<T, TTT>::value &&
                                     !std::is_same<T, TTT>::value),
                                    bool>::type = true>
  explicit operator TTT() const {
    return static_cast<TTT>(static_cast<T>(*this));
  }
};

template <typename T>
using req = required_v2<T>;

template <typename T>
using req2 = required_v2<T>;

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

} // namespace FlexFlow

#endif

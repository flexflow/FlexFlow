#ifndef _FLEXFLOW_LIB_UTILS_TEST_TYPES_INCLUDE_UTILS_TEST_TYPES_TEST_TYPES_H
#define _FLEXFLOW_LIB_UTILS_TEST_TYPES_INCLUDE_UTILS_TEST_TYPES_TEST_TYPES_H

#include "has_capability.h"
#include "utils/backports/type_identity.h"
#include <string>
#include <type_traits>

namespace FlexFlow::test_types {

template <capability_t... CAPABILITIES>
struct test_type_t {
  template <capability_t... C>
  using supports = std::conjunction<has_capability<C, CAPABILITIES...>...>;

  template <capability_t C = DEFAULT_CONSTRUCTIBLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t();

  template <capability_t C = DEFAULT_CONSTRUCTIBLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t() = delete;

  template <capability_t C = COPY_CONSTRUCTIBLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t(test_type_t const &);

  template <capability_t C = COPY_CONSTRUCTIBLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t(test_type_t const &) = delete;

  template <capability_t C = COPY_ASSIGNABLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t const &);

  template <capability_t C = COPY_ASSIGNABLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t const &) = delete;

  template <capability_t C = MOVE_CONSTRUCTIBLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t(test_type_t &&);

  template <capability_t C = MOVE_CONSTRUCTIBLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t(test_type_t &&) = delete;

  template <capability_t C = MOVE_ASSIGNABLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t &&);

  template <capability_t C = MOVE_ASSIGNABLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t &&) = delete;

  template <capability_t C = EQ>
  typename std::enable_if<supports<C>::value, bool>::type
      operator==(test_type_t const &) const;

  template <capability_t C = EQ>
  typename std::enable_if<supports<C>::value, bool>::type
      operator!=(test_type_t const &) const;

  template <capability_t C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator<(test_type_t const &) const;

  template <capability_t C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator>(test_type_t const &) const;

  template <capability_t C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator<=(test_type_t const &) const;

  template <capability_t C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator>=(test_type_t const &) const;

  template <capability_t C = PLUS>
  typename std::enable_if<supports<C>::value, test_type_t>::type
      operator+(test_type_t const &);

  template <capability_t C = PLUSEQ>
  typename std::enable_if<supports<C>::value, test_type_t>::type
      operator+=(test_type_t const &);
};

template <capability_t... CAPABILITIES>
std::enable_if_t<has_capability<FMT, CAPABILITIES...>::value, std::string>
    format_as(test_type_t<CAPABILITIES...>);

using no_eq = test_type_t<>;
using eq = test_type_t<EQ>;
using cmp = test_type_t<CMP>;
using hash_cmp = test_type_t<HASHABLE, CMP>;
using plusable = test_type_t<PLUS, PLUSEQ>;
using fmtable = test_type_t<FMT>;

template <typename T1, typename T2>
struct both;

template <capability_t... C1, capability_t... C2>
struct both<test_type_t<C1...>, test_type_t<C2...>>
    : type_identity<test_type_t<C1..., C2...>> {};

template <typename T1, typename T2>
using both_t = typename both<T1, T2>::type;

using well_behaved_value_type = test_type_t<EQ,
                                            COPY_CONSTRUCTIBLE,
                                            MOVE_CONSTRUCTIBLE,
                                            COPY_ASSIGNABLE,
                                            MOVE_ASSIGNABLE>;

using wb_hash = both_t<hash_cmp, well_behaved_value_type>;
using wb_hash_fmt = both_t<fmtable, wb_hash>;
using wb_fmt = both_t<fmtable, well_behaved_value_type>;

} // namespace FlexFlow::test_types

namespace std {

template <::FlexFlow::test_types::capability_t... CAPABILITIES>
struct hash<::FlexFlow::test_types::test_type_t<CAPABILITIES...>> {
  template <
      ::FlexFlow::test_types::capability_t C = ::FlexFlow::test_types::HASHABLE>
  typename std::enable_if<
      ::FlexFlow::test_types::has_capability<C, CAPABILITIES...>::value,
      size_t>::type
      operator()(
          ::FlexFlow::test_types::test_type_t<CAPABILITIES...> const &) const;
};

} // namespace std

#endif

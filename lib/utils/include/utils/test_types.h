#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TEST_TYPES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TEST_TYPES_H

#include "type_traits_core.h"

namespace FlexFlow {

namespace test_types {

enum capability {
  HASHABLE,
  EQ,
  CMP,
  DEFAULT_CONSTRUCTIBLE,
  MOVE_CONSTRUCTIBLE,
  MOVE_ASSIGNABLE,
  COPY_CONSTRUCTIBLE,
  COPY_ASSIGNABLE,
  PLUS,
  PLUSEQ,
  FMT
};

template <capability PRECONDITION, capability POSTCONDITION>
struct capability_implies : std::false_type {};

template <>
struct capability_implies<CMP, EQ> : std::true_type {};

template <capability C>
struct capability_implies<C, C> : std::true_type {};

template <capability NEEDLE, capability... HAYSTACK>
struct has_capability;

template <capability NEEDLE, capability HEAD, capability... HAYSTACK>
struct has_capability<NEEDLE, HEAD, HAYSTACK...>
    : disjunction<capability_implies<HEAD, NEEDLE>,
                  has_capability<NEEDLE, HAYSTACK...>> {};

template <capability NEEDLE>
struct has_capability<NEEDLE> : std::false_type {};

template <capability... CAPABILITIES>
struct test_type_t {
  template <capability... C>
  using supports = conjunction<has_capability<C, CAPABILITIES...>...>;

  template <capability C = DEFAULT_CONSTRUCTIBLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t();

  template <capability C = DEFAULT_CONSTRUCTIBLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t() = delete;

  template <capability C = COPY_CONSTRUCTIBLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t(test_type_t const &);

  template <capability C = COPY_CONSTRUCTIBLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t(test_type_t const &) = delete;

  template <capability C = COPY_ASSIGNABLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t const &);

  template <capability C = COPY_ASSIGNABLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t const &) = delete;

  template <capability C = MOVE_CONSTRUCTIBLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t(test_type_t &&);

  template <capability C = MOVE_CONSTRUCTIBLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t(test_type_t &&) = delete;

  template <capability C = MOVE_ASSIGNABLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t &&);

  template <capability C = MOVE_ASSIGNABLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t &operator=(test_type_t &&) = delete;

  template <capability C = EQ>
  typename std::enable_if<supports<C>::value, bool>::type
      operator==(test_type_t const &) const;

  template <capability C = EQ>
  typename std::enable_if<supports<C>::value, bool>::type
      operator!=(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator<(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator>(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator<=(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator>=(test_type_t const &) const;

  template <capability C = PLUS>
  typename std::enable_if<supports<C>::value, test_type_t>::type
      operator+(test_type_t const &);

  template <capability C = PLUSEQ>
  typename std::enable_if<supports<C>::value, test_type_t>::type
      operator+=(test_type_t const &);
};

template <capability... CAPABILITIES>
enable_if_t<has_capability<FMT, CAPABILITIES...>::value, std::string>
    format_as(test_type_t<CAPABILITIES...>);

using no_eq = test_type_t<>;
using eq = test_type_t<EQ>;
using cmp = test_type_t<CMP>;
using hash_cmp = test_type_t<HASHABLE, CMP>;
using plusable = test_type_t<PLUS, PLUSEQ>;
using fmtable = test_type_t<FMT>;

template <typename T1, typename T2>
struct both;

template <capability... C1, capability... C2>
struct both<test_type_t<C1...>, test_type_t<C2...>> : type_identity<test_type_t<C1..., C2...>> {};

template <typename T1, typename T2>
using both_t = typename both<T1, T2>::type;

using well_behaved_value_type = test_type_t<
  EQ, 
  COPY_CONSTRUCTIBLE,
  MOVE_CONSTRUCTIBLE,
  COPY_ASSIGNABLE,
  MOVE_ASSIGNABLE
>;

using wb_hash = both_t<hash_cmp, well_behaved_value_type>;
using wb_hash_fmt = both_t<fmtable, wb_hash>;
using wb_fmt = both_t<fmtable, well_behaved_value_type>;

} // namespace test_types
} // namespace FlexFlow

namespace std {

template <
    ::FlexFlow::test_types::
        capability... CAPABILITIES> //, typename = typename
                                    // std::enable_if<::FlexFlow::test_types::has_capability<::FlexFlow::test_types::HASHABLE>::value,
                                    // bool>::type>
struct hash<::FlexFlow::test_types::test_type_t<CAPABILITIES...>> {
  template <::FlexFlow::test_types::capability C = ::FlexFlow::test_types::HASHABLE>
  typename std::enable_if<
      ::FlexFlow::test_types::has_capability<C, CAPABILITIES...>::value,
      size_t>::type
      operator()(
          ::FlexFlow::test_types::test_type_t<CAPABILITIES...> const &) const;
};

} // namespace std

#endif

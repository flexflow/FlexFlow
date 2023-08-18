#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TEST_TYPES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TEST_TYPES_H

#include "type_traits.h"

namespace FlexFlow {

namespace test_types {

enum capability { HASHABLE, EQ, CMP, DEFAULT_CONSTRUCTIBLE, COPYABLE };

template <capability PRECONDITION, capability POSTCONDITION>
struct capability_implies : std::false_type {};

template <>
struct capability_implies<EQ, CMP> : std::true_type {};

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

  template <typename std::enable_if<supports<DEFAULT_CONSTRUCTIBLE>::value,
                                    bool>::type = true>
  test_type_t();

  template <typename std::enable_if<!supports<DEFAULT_CONSTRUCTIBLE>::value,
                                    bool>::type = true>
  test_type_t() = delete;

  template <
      typename std::enable_if<supports<COPYABLE>::value, bool>::type = true>
  test_type_t(test_type_t const &);

  template <
      typename std::enable_if<!supports<COPYABLE>::value, bool>::type = true>
  test_type_t(test_type_t const &) = delete;

  typename std::enable_if<supports<EQ>::value, bool>::type
      operator==(test_type_t const &) const;

  typename std::enable_if<supports<EQ>::value, bool>::type
      operator!=(test_type_t const &) const;

  typename std::enable_if<supports<CMP>::value, bool>::type
      operator<(test_type_t const &) const;

  typename std::enable_if<supports<CMP>::value, bool>::type
      operator>(test_type_t const &) const;

  typename std::enable_if<supports<CMP>::value, bool>::type
      operator<=(test_type_t const &) const;

  typename std::enable_if<supports<CMP>::value, bool>::type
      operator>=(test_type_t const &) const;
};

using no_eq = test_type_t<>;
using eq = test_type_t<EQ>;
using cmp = test_type_t<CMP>;
using hash_cmp = test_type_t<HASHABLE, CMP>;

} // namespace test_types
} // namespace FlexFlow

namespace std {

template <::FlexFlow::test_types::capability... CAPABILITIES>
struct hash<::FlexFlow::test_types::test_type_t<CAPABILITIES...>> {
  typename std::enable_if<
      ::FlexFlow::test_types::has_capability<::FlexFlow::test_types::HASHABLE,
                                             CAPABILITIES...>::value,
      size_t>::type
      operator()(
          ::FlexFlow::test_types::test_type_t<CAPABILITIES...> const &) const;
};

} // namespace std

#endif

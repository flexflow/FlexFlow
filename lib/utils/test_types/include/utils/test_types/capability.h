#ifndef _FLEXFLOW_LIB_UTILS_TEST_TYPES_INCLUDE_UTILS_TEST_TYPES_CAPABILITY_H
#define _FLEXFLOW_LIB_UTILS_TEST_TYPES_INCLUDE_UTILS_TEST_TYPES_CAPABILITY_H

#include <string>

namespace FlexFlow::test_types {

enum capability_t {
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

std::string format_as(capability_t c);

template <capability_t PRECONDITION, capability_t POSTCONDITION>
struct capability_implies : std::false_type {};

template <>
struct capability_implies<CMP, EQ> : std::true_type {};

template <capability_t C>
struct capability_implies<C, C> : std::true_type {};

} // namespace FlexFlow::test_types

#endif

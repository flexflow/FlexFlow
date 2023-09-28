#ifndef _FLEXFLOW_LIB_UTILS_TEST_TYPES_INCLUDE_UTILS_TEST_TYPES_HAS_CAPABILITY_H
#define _FLEXFLOW_LIB_UTILS_TEST_TYPES_INCLUDE_UTILS_TEST_TYPES_HAS_CAPABILITY_H

#include "capability.h"

namespace FlexFlow::test_types {

template <capability_t NEEDLE, capability_t... HAYSTACK>
struct has_capability;

template <capability_t NEEDLE, capability_t HEAD, capability_t... HAYSTACK>
struct has_capability<NEEDLE, HEAD, HAYSTACK...>
    : std::disjunction<capability_implies<HEAD, NEEDLE>,
                  has_capability<NEEDLE, HAYSTACK...>> {};

template <capability_t NEEDLE>
struct has_capability<NEEDLE> : std::false_type {};

} // namespace FlexFlow

#endif

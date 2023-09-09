#ifndef _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IMPLIES_H
#define _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IMPLIES_H

#include <type_traits>

namespace FlexFlow {

template <typename LHS, typename RHS>
struct implies : std::disjunction<RHS, std::negation<LHS>> {};

template <typename LHS, typename RHS>
inline constexpr bool implies_v = implies<LHS, RHS>::value;

}

#endif

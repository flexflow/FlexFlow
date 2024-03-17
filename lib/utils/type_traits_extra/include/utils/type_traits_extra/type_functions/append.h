#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_FUNCTIONS_APPEND_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_FUNCTIONS_APPEND_H

namespace FlexFlow {

template <typename List, typename Element>
struct append {};

template <typename List, typename Element>
using append_t = typename append<List, Element>::type;

} // namespace FlexFlow

#endif

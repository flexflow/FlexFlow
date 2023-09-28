#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_ITERATOR_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_ITERATOR_H

#include <type_traits>
#include <iterator>
#include "nameof.hpp"
#include "debug_print_type.h"
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename C, typename Tag>
struct supports_iterator_tag
    : std::is_base_of<Tag,
                      typename std::iterator_traits<
                          typename C::iterator>::iterator_category> {};

template <typename C, typename Tag>
inline constexpr bool supports_iterator_tag_v = supports_iterator_tag<C, Tag>::value;

#define CHECK_SUPPORTS_ITERATOR_TAG(TAG, ...)                                  \
  static_assert(supports_iterator_tag_v<__VA_ARGS__, TAG>,                \
       #__VA_ARGS__ " does not support required iterator tag " #TAG); \
  ERROR_PRINT_TYPE(WRAP_ARG(supports_iterator_tag_v<__VA_ARGS__, TAG>), "container_type", __VA_ARGS__); \

using TTT = std::vector<int>;
CHECK_SUPPORTS_ITERATOR_TAG(std::random_access_iterator_tag, TTT);
CHECK_SUPPORTS_ITERATOR_TAG(std::random_access_iterator_tag, TTT);


}

#endif

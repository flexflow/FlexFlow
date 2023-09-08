#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_ITERATOR_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_ITERATOR_H

#include <type_traits>
#include <iterator>

namespace FlexFlow {

template <typename C, typename Tag>
struct supports_iterator_tag
    : std::is_base_of<Tag,
                      typename std::iterator_traits<
                          typename C::iterator>::iterator_category> {};

#define CHECK_SUPPORTS_ITERATOR_TAG(TAG, ...)                                  \
  static_assert(supports_iterator_tag<__VA_ARGS__, TAG>::value,                \
                #__VA_ARGS__ " does not support required iterator tag " #TAG);


}

#endif

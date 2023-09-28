#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_CHECK_ABSTRACT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_CHECK_ABSTRACT_H

#include <type_traits>

#define CHECK_NOT_ABSTRACT(...)                                                \
  static_assert(                                                               \
      !std::is_abstract<__VA_ARGS__>::value,                                   \
      #__VA_ARGS__                                                             \
      " should not be abstract (are you missing a virtual method override?)");

#endif

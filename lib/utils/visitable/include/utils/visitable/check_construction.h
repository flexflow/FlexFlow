#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_CONSTRUCTION_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_CONSTRUCTION_H

#include "utils/preprocessor_extra/wrap_arg.h"
#include "utils/visitable/is_only_visit_list_initializable.h"

#define CHECK_CONSTRUCTION_NONEMPTY(...)                           \
  static_assert \
  (is_only_visit_list_initializable<__VA_ARGS__>::value,              \
   #__VA_ARGS__ " should not be list-initialializable from any sub-tuples "   \
                 "(you probably need to insert req<...>s)");                   \
  static_assert(!std::is_default_constructible<__VA_ARGS__>::value,                       \
        #__VA_ARGS__ " should not be default-constructible (you "                 \
                  "probably need to insert req<...>s)");                       \
  static_assert(is_visit_list_initializable<__VA_ARGS__>::value,                          \
        #__VA_ARGS__                                                              \
        " should be list-initialializable by the visit field types");

#define _CHECK_CONSTRUCTION_EMPTY(...)                             \
  static_assert(std::is_default_constructible<__VA_ARGS__>::value,                        \
        #__VA_ARGS__ " should be default-constructible as it is empty")

#endif

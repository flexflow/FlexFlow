#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_MAKE_VISIT_HASHABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_MAKE_VISIT_HASHABLE_H

#include "utils/visitable/use_visitable_hash.h"
#include <functional>

#define MAKE_VISIT_HASHABLE(TYPENAME)                                          \
  namespace std {                                                              \
  template <>                                                                  \
  struct hash<TYPENAME> : ::FlexFlow::use_visitable_hash<TYPENAME> {};         \
  }                                                                            \
  static_assert(true, "")

#endif

#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_FF_VISITABLE_STRUCT_DEFAULT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_FF_VISITABLE_STRUCT_DEFAULT_H

#include "utils/visitable/dispatch_visitable.h"
#include "utils/visitable/make_visit_hashable.h"
#include "utils/visitable/visitable_struct_empty.h"
#include "visit_struct/visit_struct.hpp"
#include "utils/visitable/check_well_behaved_visit_type.h"
#include "utils/visitable/check_construction.h"

#define FF_VISITABLE_STRUCT_EMPTY(TYPENAME)                                    \
  }                                                                            \
  VISITABLE_STRUCT_EMPTY(::FlexFlow::TYPENAME);                                \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE(TYPENAME);                                     \
  CHECK_CONSTRUCTION_EMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT_NONEMPTY(TYPENAME, ...)                            \
  }                                                                            \
  VISITABLE_STRUCT(::FlexFlow::TYPENAME, __VA_ARGS__);                         \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE(TYPENAME);                                     \
  CHECK_CONSTRUCTION_NONEMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT(...)                                               \
  _DISPATCH_VISITABLE(FF_VISITABLE_STRUCT, __VA_ARGS__)

#endif

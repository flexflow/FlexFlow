#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION_H

#define FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION_EMPTY(TYPENAME)           \
  }                                                                            \
  VISITABLE_STRUCT_EMPTY(::FlexFlow::TYPENAME);                                \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME);

#define FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION_NONEMPTY(TYPENAME, ...)   \
  }                                                                            \
  VISITABLE_STRUCT(::FlexFlow::TYPENAME, __VA_ARGS__);                         \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME);

#define FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(...)                      \
  _DISPATCH_VISITABLE(FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION, __VA_ARGS__)

#endif

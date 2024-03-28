#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_FF_VISITABLE_STRUCT_NO_EQ_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_FF_VISITABLE_STRUCT_NO_EQ_H

#define FF_VISITABLE_STRUCT_NO_EQ_EMPTY(TYPENAME)                              \
  }                                                                            \
  VISITABLE_STRUCT_EMPTY(::FlexFlow::TYPENAME);                                \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_CONSTRUCTION_EMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT_NO_EQ_NONEMPTY(TYPENAME, ...)                      \
  }                                                                            \
  VISITABLE_STRUCT(::FlexFlow::TYPENAME, __VA_ARGS__);                         \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_CONSTRUCTION_NONEMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT_NO_EQ(...)                                         \
  _DISPATCH_VISITABLE(FF_VISITABLE_STRUCT_NO_EQ, __VA_ARGS__)

#endif

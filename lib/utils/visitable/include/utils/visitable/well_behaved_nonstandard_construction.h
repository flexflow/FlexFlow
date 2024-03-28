#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_WELL_BEHAVED_NONSTANDARD_CONSTRUCTION_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_WELL_BEHAVED_NONSTANDARD_CONSTRUCTION_H

namespace FlexFlow {

#define CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION_NO_EQ(TYPENAME) \
  static_assert(is_visitable<TYPENAME>::value,                                 \
                #TYPENAME " is not visitable (this should never "              \
                          "happen--contact the FF developers)");               \
  static_assert(sizeof(visit_as_tuple_raw_t<TYPENAME>) == sizeof(TYPENAME),    \
                #TYPENAME " should be fully visitable");                       \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(TYPENAME);

#define CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME)       \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION_NO_EQ(TYPENAME);      \
  CHECK_WELL_BEHAVED_VALUE_TYPE(TYPENAME);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_WELL_BEHAVED_NONSTANDARD_CONSTRUCTION_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_WELL_BEHAVED_NONSTANDARD_CONSTRUCTION_H

namespace FlexFlow {

#define CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION_NO_EQ(...) \
  static_assert(is_visitable<__VA_ARGS__>::value,                                 \
                #__VA_ARGS__ " is not visitable (this should never "              \
                          "happen--contact the FF developers)");               \
  static_assert(sizeof(visit_as_tuple_raw_t<__VA_ARGS__>) == sizeof(TYPENAME),    \
                #__VA_ARGS__ " should be fully visitable");                       \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(__VA_ARGS__);

#define CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(...)       \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION_NO_EQ(__VA_ARGS__);      \
  CHECK_WELL_BEHAVED_VALUE_TYPE(__VA_ARGS__);

} // namespace FlexFlow

#endif

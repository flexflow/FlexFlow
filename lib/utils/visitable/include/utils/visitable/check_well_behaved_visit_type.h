#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_WELL_BEHAVED_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_WELL_BEHAVED_H

#define CHECK_WELL_BEHAVED_VISIT_TYPE(TYPENAME)                    \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME);            \
  static_assert(is_visit_list_initializable_v<TYPENAME>,                          \
             #TYPENAME                                                         \
             " should be list-initialializable by the visit field types");

#endif

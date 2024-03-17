#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_CONSTRUCTION_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_CHECKS_CONSTRUCTION_H

#define CHECK_CONSTRUCTION_NONEMPTY(CHECK, TYPENAME)                           \
  UNWRAP_ARG(CHECK)                                                            \
  (is_only_visit_list_initializable<UNWRAP_ARG(TYPENAME)>::value,              \
   #UNWRAP_ARG(                                                                \
       TYPENAME) " should not be list-initialializable from any sub-tuples "   \
                 "(you probably need to insert req<...>s)");                   \
  CHECK(!std::is_default_constructible<TYPENAME>::value,                       \
        #TYPENAME " should not be default-constructible (you "                 \
                  "probably need to insert req<...>s)");                       \
  CHECK(is_visit_list_initializable<TYPENAME>::value,                          \
        #TYPENAME                                                              \
        " should be list-initialializable by the visit field types");

#define _CHECK_CONSTRUCTION_EMPTY(CHECK, TYPENAME)                             \
  CHECK(std::is_default_constructible<TYPENAME>::value,                        \
        #TYPENAME " should be default-constructible as it is empty")

#endif

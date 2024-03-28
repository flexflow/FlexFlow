#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_VISITABLE_STRUCT_EMPTY_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_VISITABLE_STRUCT_EMPTY_H

#include "visit_struct/visit_struct.hpp"

#define VISITABLE_STRUCT_EMPTY(STRUCT_NAME)                                    \
  namespace visit_struct {                                                     \
  namespace traits {                                                           \
                                                                               \
  template <>                                                                  \
  struct visitable<STRUCT_NAME, void> {                                        \
                                                                               \
    using this_type = STRUCT_NAME;                                             \
                                                                               \
    static VISIT_STRUCT_CONSTEXPR auto get_name() -> decltype(#STRUCT_NAME) {  \
      return #STRUCT_NAME;                                                     \
    }                                                                          \
                                                                               \
    static VISIT_STRUCT_CONSTEXPR const std::size_t field_count = 0;           \
                                                                               \
    template <typename V, typename S>                                          \
    VISIT_STRUCT_CXX14_CONSTEXPR static void apply(V &&visitor,                \
                                                   S &&struct_instance) {}     \
                                                                               \
    template <typename V, typename S1, typename S2>                            \
    VISIT_STRUCT_CXX14_CONSTEXPR static void                                   \
        apply(V &&visitor, S1 &&s1, S2 &&s2) {}                                \
                                                                               \
    template <typename V>                                                      \
    VISIT_STRUCT_CXX14_CONSTEXPR static void visit_pointers(V &&visitor) {}    \
                                                                               \
    template <typename V>                                                      \
    VISIT_STRUCT_CXX14_CONSTEXPR static void visit_types(V &&visitor) {}       \
                                                                               \
    template <typename V>                                                      \
    VISIT_STRUCT_CXX14_CONSTEXPR static void visit_accessors(V &&visitor) {}   \
                                                                               \
    struct fields_enum {                                                       \
      enum index {};                                                           \
    };                                                                         \
                                                                               \
    static VISIT_STRUCT_CONSTEXPR const bool value = true;                     \
  };                                                                           \
  }                                                                            \
  }                                                                            \
  static_assert(true, "")

#endif

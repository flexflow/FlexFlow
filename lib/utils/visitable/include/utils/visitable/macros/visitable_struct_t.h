#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_VISITABLE_STRUCT_T_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_VISITABLE_STRUCT_T_H

#include "utils/preprocessor_extra/template.h"
#include "visit_struct/visit_struct.hpp"

#define VISITABLE_STRUCT_T(STRUCT_NAME, NT, ...)                                                         \
namespace visit_struct {                                                                           \
namespace traits {                                                                                 \
                                                                                                   \
template <TEMPLATE_DECL(NT)>                                                                                        \
struct visitable< TEMPLATE_SPECIALIZE(STRUCT_NAME, NT) , void> {                                                              \
                                                                                                   \
  using this_type = TEMPLATE_SPECIALIZE(STRUCT_NAME, NT);                                                                   \
                                                                                                   \
  static VISIT_STRUCT_CONSTEXPR auto get_name()                                                    \
    -> decltype(#STRUCT_NAME) {                                                                    \
    return #STRUCT_NAME;                                                                           \
  }                                                                                                \
                                                                                                   \
  static VISIT_STRUCT_CONSTEXPR const std::size_t field_count = 0                                  \
    VISIT_STRUCT_PP_MAP(VISIT_STRUCT_FIELD_COUNT, __VA_ARGS__);                                    \
                                                                                                   \
  template <typename V, typename S>                                                                \
  VISIT_STRUCT_CXX14_CONSTEXPR static void apply(V && visitor, S && struct_instance)               \
  {                                                                                                \
    VISIT_STRUCT_PP_MAP(VISIT_STRUCT_MEMBER_HELPER, __VA_ARGS__)                                   \
  }                                                                                                \
                                                                                                   \
  template <typename V, typename S1, typename S2>                                                  \
  VISIT_STRUCT_CXX14_CONSTEXPR static void apply(V && visitor, S1 && s1, S2 && s2)                 \
  {                                                                                                \
    VISIT_STRUCT_PP_MAP(VISIT_STRUCT_MEMBER_HELPER_PAIR, __VA_ARGS__)                              \
  }                                                                                                \
                                                                                                   \
  template <typename V>                                                                            \
  VISIT_STRUCT_CXX14_CONSTEXPR static void visit_pointers(V && visitor)                            \
  {                                                                                                \
    VISIT_STRUCT_PP_MAP(VISIT_STRUCT_MEMBER_HELPER_PTR, __VA_ARGS__)                               \
  }                                                                                                \
                                                                                                   \
  template <typename V>                                                                            \
  VISIT_STRUCT_CXX14_CONSTEXPR static void visit_types(V && visitor)                               \
  {                                                                                                \
    VISIT_STRUCT_PP_MAP(VISIT_STRUCT_MEMBER_HELPER_TYPE, __VA_ARGS__)                              \
  }                                                                                                \
                                                                                                   \
  template <typename V>                                                                            \
  VISIT_STRUCT_CXX14_CONSTEXPR static void visit_accessors(V && visitor)                           \
  {                                                                                                \
    VISIT_STRUCT_PP_MAP(VISIT_STRUCT_MEMBER_HELPER_ACC, __VA_ARGS__)                               \
  }                                                                                                \
                                                                                                   \
  struct fields_enum {                                                                             \
    enum index { __VA_ARGS__ };                                                                    \
  };                                                                                               \
                                                                                                   \
  VISIT_STRUCT_PP_MAP(VISIT_STRUCT_MAKE_GETTERS, __VA_ARGS__)                                      \
                                                                                                   \
  static VISIT_STRUCT_CONSTEXPR const bool value = true;                                           \
};                                                                                                 \
                                                                                                   \
}                                                                                                  \
}                                                                                                  \
static_assert(true, "")

#endif

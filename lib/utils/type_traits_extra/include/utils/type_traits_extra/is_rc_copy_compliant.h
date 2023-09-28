#ifndef _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_RC_COPY_COMPLIANT_H
#define _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_RC_COPY_COMPLIANT_H

namespace FlexFlow {

#define RC_COPY_VIRTUAL_MSG                                                    \
  "https://isocpp.github.io/CppCoreGuidelines/"                                \
  "CppCoreGuidelines#Rc-copy-virtual"

#define CHECK_RC_COPY_VIRTUAL_COMPLIANT(...)                                   \
  static_assert(                                                               \
      !std::is_copy_constructible<__VA_ARGS__>::value,                         \
      #__VA_ARGS__                                                             \
      " should not be copy-constructible. See " RC_COPY_VIRTUAL_MSG);          \
  static_assert(!std::is_copy_assignable<__VA_ARGS__>::value,                  \
                #__VA_ARGS__                                                   \
                " should not be copy-assignable. See " RC_COPY_VIRTUAL_MSG);   \
  static_assert(                                                               \
      !std::is_move_constructible<__VA_ARGS__>::value,                         \
      #__VA_ARGS__                                                             \
      " should not be move-constructible. See " RC_COPY_VIRTUAL_MSG);          \
  static_assert(!std::is_move_assignable<__VA_ARGS__>::value,                  \
                #__VA_ARGS__                                                   \
                " should not be move-assignable. See " RC_COPY_VIRTUAL_MSG);   \
  static_assert(std::has_virtual_destructor<__VA_ARGS__>::value,               \
                #__VA_ARGS__                                                   \
                " should have a virtual destructor. See " RC_COPY_VIRTUAL_MSG)

template <typename T>
struct is_rc_copy_virtual_compliant
    : std::conjunction<std::negation<std::disjunction<std::is_copy_constructible<T>,
                                       std::is_copy_assignable<T>,
                                       std::is_move_constructible<T>,
                                       std::is_move_assignable<T>>>,
                  std::has_virtual_destructor<T>> {};


template <typename T>
inline constexpr bool is_rc_copy_virtual_compliant_v =
    is_rc_copy_virtual_compliant<T>::value;

}

#endif

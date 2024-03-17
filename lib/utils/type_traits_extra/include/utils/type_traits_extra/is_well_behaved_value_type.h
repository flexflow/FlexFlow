#ifndef _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_WELL_BEHAVED_VALUE_TYPE_H
#define _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_WELL_BEHAVED_VALUE_TYPE_H

#include "is_equal_comparable.h"
#include "is_hashable.h"
#include "is_neq_comparable.h"
#include "operators.h"
#include <type_traits>

namespace FlexFlow {

template <typename T>
struct is_well_behaved_value_type_no_hash
    : std::conjunction<is_equal_comparable<T>,
                       is_neq_comparable<T>,
                       std::is_copy_constructible<T>,
                       std::is_move_constructible<T>,
                       std::is_copy_assignable<T>,
                       std::is_move_assignable<T>> {};

#define CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(...)                               \
  static_assert(is_copy_constructible<__VA_ARGS__>::value,                     \
                #__VA_ARGS__ " should be copy-constructible");                 \
  static_assert(is_move_constructible<__VA_ARGS__>::value,                     \
                #__VA_ARGS__ " should be move-constructible");                 \
  static_assert(is_copy_assignable<__VA_ARGS__>::value,                        \
                #__VA_ARGS__ " should be copy-assignable");                    \
  static_assert(is_move_assignable<__VA_ARGS__>::value,                        \
                #__VA_ARGS__ " should be move-assignable")

#define CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(...)                             \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(__VA_ARGS__);                            \
  static_assert(is_equal_comparable<__VA_ARGS__>::value,                       \
                #__VA_ARGS__ " should support operator==");                    \
  static_assert(is_neq_comparable<__VA_ARGS__>::value,                         \
                #__VA_ARGS__ " should support operator!=");

template <typename T>
struct is_well_behaved_value_type
    : std::conjunction<is_well_behaved_value_type_no_hash<T>, is_hashable<T>> {
};

template <typename T>
inline constexpr bool is_well_behaved_value_type_v =
    is_well_behaved_value_type<T>::value;

#define CHECK_WELL_BEHAVED_VALUE_TYPE(...)                                     \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(__VA_ARGS__);                          \
  CHECK_HASHABLE(__VA_ARGS__)

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H

#include "utils/fmt.h"
#include "utils/invoke.h"
#include "utils/metafunction.h"
#include "utils/type_traits.decl.h"
#include "utils/type_traits_core.h"
#include "utils/visitable_core.h"
#include <iostream>
#include <type_traits>
#include "utils/variant.h"

namespace FlexFlow {

template <typename T, typename Enable>
struct is_clonable : std::false_type {};

template <typename T>
struct is_clonable<T, void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T, void_t<decltype(std::cout << std::declval<T>())>>
    : std::true_type {};

template <typename T, typename Enable>
struct is_lt_comparable : std::false_type {};

template <typename T>
struct is_lt_comparable<
    T,
    void_t<decltype((bool)(std::declval<T>() < std::declval<T>()))>>
    : std::true_type {};


template <typename... Ts>
struct types_are_all_same : std::false_type {};

template <>
struct types_are_all_same<> : std::true_type {};

template <typename T>
struct types_are_all_same<T> : std::true_type {};

template <typename Head, typename Next, typename... Rest>
struct types_are_all_same<Head, Next, Rest...>
    : conjunction<std::is_same<Head, Next>, types_are_all_same<Head, Rest...>> {
};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_TUPLE_H
#define _FLEXFLOW_UTILS_TUPLE_H

#include <cstddef>
#include <tuple>
#include <type_traits>

// Adapted from
// https://github.com/bitwizeshift/BackportCpp/blob/4f33a7f9b219f169e60d8ed2fd5731a3a23288e4/include/bpstd/tuple.hpp

namespace FlexFlow {

namespace TupleUtils {

template <typename T, std::size_t Index, typename... Types>
struct index_of_impl;

template <typename T, std::size_t Index, typename Type0, typename... Types>
struct index_of_impl<T, Index, Type0, Types...>
    : index_of_impl<T, Index + 1, Types...> {};

template <typename T, std::size_t Index, typename... Types>
struct index_of_impl<T, Index, T, Types...>
    : std::integral_constant<std::size_t, Index> {};

template <typename T, typename... Types>
struct index_of : index_of_impl<T, 0, Types...> {};

}; // namespace TupleUtils

template <typename T, typename... Types>
T &get(std::tuple<Types...> &t) noexcept {
  return std::get<TupleUtils::index_of<T, Types...>::value>(t);
}

template <typename T, typename... Types>
T &&get(std::tuple<Types...> &&t) noexcept {
  return move(std::get<TupleUtils::index_of<T, Types...>::value>(t));
}

template <typename T, typename... Types>
T const &get(std::tuple<Types...> const &t) noexcept {
  return std::get<TupleUtils::index_of<T, Types...>::value>(t);
}

template <typename T, typename... Types>
T const &&get(std::tuple<Types...> const &&t) noexcept {
  return move(std::get<TupleUtils::index_of<T, Types...>::value>(t));
}

}; // namespace FlexFlow

#endif // _FLEXFLOW_UTILS_TUPLE_H
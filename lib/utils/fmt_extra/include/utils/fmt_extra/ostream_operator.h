#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_OSTREAM_OPERATOR_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_OSTREAM_OPERATOR_H

#include "utils/fmt_extra/is_fmtable.h"
#include <fmt/format.h>
#include <string>
#include <type_traits>

namespace FlexFlow {

template <typename T>
struct ostream_operator_delegate_is_expected : std::false_type {};

template <typename T>
inline constexpr bool ostream_operator_delegate_is_expected_v =
    ostream_operator_delegate_is_expected<T>::value;

template <typename T, typename Enable = void>
struct do_not_delegate_ostream_operator : std::false_type {};

template <typename T>
inline constexpr bool do_not_delegate_ostream_operator_v =
    do_not_delegate_ostream_operator<T>::value;

template <>
struct do_not_delegate_ostream_operator<int> : std::true_type {};

template <>
struct do_not_delegate_ostream_operator<char> : std::true_type {};

template <>
struct do_not_delegate_ostream_operator<std::string> : std::true_type {};

template <std::size_t N>
struct do_not_delegate_ostream_operator<char[N]> : std::true_type {};

template <>
struct do_not_delegate_ostream_operator<char const *> : std::true_type {};

template <typename T>
std::enable_if_t<is_fmtable_v<T> && !do_not_delegate_ostream_operator_v<T> &&
                     !ostream_operator_delegate_is_expected_v<T>,
                 std::ostream &>
    operator<<(std::ostream &s, T const &t) {

  std::string result = fmt::to_string(t);
  return s << result;
}

template <typename T>
std::enable_if_t<ostream_operator_delegate_is_expected_v<T>, std::ostream &>
    operator<<(std::ostream &s, T const &t) {
  static_assert(is_fmtable_v<T>, "Expected type to be fmtable");

  std::string result = fmt::to_string(t);
  return s << result;
}

} // namespace FlexFlow

#endif

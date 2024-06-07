#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "utils/containers.h"
#include "utils/fmt.decl.h"
#include "utils/test_types.h"
#include "utils/type_traits_core.h"
#include <unordered_set>
#include <variant>
#include <vector>
#include <iomanip>

namespace FlexFlow {

template <typename T, typename Enable>
struct already_has_ostream_operator : std::false_type {};

template <> struct already_has_ostream_operator<int> : std::true_type {};

template <> struct already_has_ostream_operator<char> : std::true_type {};

template <>
struct already_has_ostream_operator<std::string> : std::true_type {};

template <size_t N>
struct already_has_ostream_operator<char[N]> : std::true_type {};

template <>
struct already_has_ostream_operator<char const *> : std::true_type {};

template <>
struct already_has_ostream_operator<std::_Setfill<char>> : std::true_type {};

template <> struct already_has_ostream_operator<std::_Setw> : std::true_type {};

// This will create an error
/*
template <typename T>
std::ostream &
operator<<(std::ostream &s, T const &t) {
  return s << "FlexFlow::ostream<<";
}
*/

#define CHECK_FMTABLE(...)                                                     \
  static_assert(::FlexFlow::is_fmtable<__VA_ARGS__>::value,                    \
                #__VA_ARGS__ " must be fmtable");

// This will not
template <typename T>
typename std::enable_if<!already_has_ostream_operator<T>::value,
                        std::ostream &>::type
operator<<(std::ostream &s, T const &t) {
  CHECK_FMTABLE(T);
  std::string result = fmt::to_string(t);
  return s << result;
}

// template <typename T>
// typename std::enable_if<is_fmtable<T>::value, std::ostream &>::type
// operator<<(std::ostream &s, T const &t) {
//     return s << fmt::to_string(t);
// }

} // namespace FlexFlow

namespace fmt {

template <typename T, typename Char>
template <typename FormatContext>
auto formatter<
    ::std::unordered_set<T>,
    Char,
    std::enable_if_t<!detail::has_format_as<std::unordered_set<T>>::value>>::
    format(::std::unordered_set<T> const &m, FormatContext &ctx)
        -> decltype(ctx.out()) {
  /* CHECK_FMTABLE(T); */

  /* std::string result = ::FlexFlow::join_strings( */
  /*     m.cbegin(), m.cend(), ", ", [](T const &t) { return fmt::to_string(t);
   * }); */
  std::string result = "";
  return formatter<std::string>::format(result, ctx);
}

template <typename T, typename Char>
template <typename FormatContext>
auto formatter<
    ::std::vector<T>,
    Char,
    std::enable_if_t<!detail::has_format_as<std::vector<T>>::value>>::
    format(::std::vector<T> const &m, FormatContext &ctx)
        -> decltype(ctx.out()) {
  CHECK_FMTABLE(T);

  std::string result = ::FlexFlow::join_strings(
      m.cbegin(), m.cend(), ", ", [](T const &t) { return fmt::to_string(t); });
  return formatter<std::string>::format("[" + result + "]", ctx);
}

template <typename... Ts>
template <typename FormatContext>
auto formatter<::std::variant<Ts...>>::format(::std::variant<Ts...> const &m,
                                              FormatContext &ctx)
    -> decltype(ctx.out()) {

  std::string result =
      std::visit([](auto &&x) { return fmt::to_string(x); }, m);
  return formatter<std::string>::format(result, ctx);
}
} // namespace fmt

namespace FlexFlow {

template <typename T>
struct delegate_ostream_operator<std::vector<T>> : std::true_type {};

template <typename T>
struct delegate_ostream_operator<std::unordered_set<T>> : std::true_type {};

template <typename T1, typename T2>
struct delegate_ostream_operator<std::pair<T1, T2>> : std::true_type {};

template <typename T>
struct delegate_ostream_operator<std::optional<T>> : std::true_type {};

template <typename... Ts>
struct delegate_ostream_operator<std::variant<Ts...>> : std::true_type {};

template <typename T>
typename std::enable_if<delegate_ostream_operator<std::decay_t<T>>::value,
                        std::ostream &>::type
    operator<<(std::ostream &s, T t) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(t);
}

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "fmt/core.h"
#include "fmt/format.h"
#include "utils/containers.decl"
#include "utils/type_traits_core.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_fmtable : std::false_type {};

template <typename T>
struct is_fmtable<T, void_t<decltype(fmt::format("{}", std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct already_has_ostream_operator : std::false_type {};

template <>
struct already_has_ostream_operator<int> : std::true_type {};

template <>
struct already_has_ostream_operator<char> : std::true_type {};

template <>
struct already_has_ostream_operator<std::string> : std::true_type {};

template <size_t N>
struct already_has_ostream_operator<char[N]> : std::true_type {};

template <>
struct already_has_ostream_operator<char const *> : std::true_type {};

// This will create an error
/*
template <typename T>
std::ostream &
operator<<(std::ostream &s, T const &t) {
  return s << "FlexFlow::ostream<<";
}
*/

// This will not
template <typename T>
typename std::enable_if<!already_has_ostream_operator<T>::value,
                        std::ostream &>::type
    operator<<(std::ostream &s, T const &t) {
  std::string result = fmt::format("{}", t);
  return s << result;
}

// template <typename T>
// typename std::enable_if<is_fmtable<T>::value, std::ostream &>::type
// operator<<(std::ostream &s, T const &t) {
//     return s << fmt::to_string(t);
// }

} // namespace FlexFlow

namespace fmt {

template <typename T>
struct formatter<::std::unordered_set<T>> : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::unordered_set<T> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    std::string result =
        join_strings(m.cbegin(), m.cend(), ", ", [](T const &t) {
          return fmt::to_string(t);
        });
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

#endif

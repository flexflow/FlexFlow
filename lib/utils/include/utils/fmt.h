#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "utils/containers.h"
#include "utils/fmt.decl.h"
#include "utils/test_types.h"
#include "utils/type_traits_core.h"
#include <variant>
#include <iomanip>

namespace FlexFlow {

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

namespace fmt {

template <typename T>
template <typename FormatContext>
auto formatter<::std::unordered_set<T>>::format(
    ::std::unordered_set<T> const &m, FormatContext &ctx)
    -> decltype(ctx.out()) {
  CHECK_FMTABLE(T);

  std::string result = ::FlexFlow::join_strings(
      m.cbegin(), m.cend(), ", ", [](T const &t) { return fmt::to_string(t); });
  return formatter<std::string>::format(result, ctx);
}

template <typename T>
template <typename FormatContext>
auto formatter<::std::vector<T>>::format(::std::vector<T> const &m,
                                         FormatContext &ctx)
    -> decltype(ctx.out()) {
  CHECK_FMTABLE(T);

  std::string result = ::FlexFlow::join_strings(
      m.cbegin(), m.cend(), ", ", [](T const &t) { return fmt::to_string(t); });
  return formatter<std::string>::format(result, ctx);
}

template <typename... Ts>
template <typename FormatContext>
auto formatter<::std::variant<Ts...>>::format(::std::variant<Ts...> const &m,
                                              FormatContext &ctx)
    -> decltype(ctx.out()) {

  std::string result = std::visit([](auto &&x) { return fmt::to_string(x); }, m);
  return formatter<std::string>::format(result, ctx);
}

} // namespace fmt

#endif

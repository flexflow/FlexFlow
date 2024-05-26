#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "utils/containers.h"
#include "utils/fmt.decl.h"
#include "utils/test_types.h"
#include "utils/type_traits_core.h"
#include <iomanip>
#include <unordered_set>
#include <variant>
#include <vector>

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

/* template <typename T> */
/* std::string format_as(::std::unordered_set<T> const &m) { */
/*   return::string result = ::FlexFlow::join_strings( */
/*       m.cbegin(), m.cend(), ", ", [](T const &t) { return fmt::to_string(t);
 * }); */
/* } */

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

/* template <typename L, typename R, typename Char> */
/* template <typename FormatContext> */
/* auto formatter< */
/*   ::std::pair<L, R>, */
/*   Char, */
/*   std::enable_if_t<!detail::has_format_as<::std::pair<L, R>>::value> */
/* >::format(::std::pair<L, R> const &m, FormatContext &ctx) */
/*     -> decltype(ctx.out()) { */
/*   /1* CHECK_FMTABLE(T); *1/ */

/*   /1* std::string result = ::FlexFlow::join_strings( *1/ */
/*   /1*     m.cbegin(), m.cend(), ", ", [](T const &t) { return
 * fmt::to_string(t); }); *1/ */
/*   NOT_IMPLEMENTED(); */
/*   std::string result = ""; */
/*   return formatter<std::string>::format(result, ctx); */
/* } */

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

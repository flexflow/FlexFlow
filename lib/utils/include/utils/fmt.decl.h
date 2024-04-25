#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H

#include "fmt/format.h"
#include <unordered_set>
#include <vector>
#include <variant>
#include "utils/check_fmtable.h"
#include <utility>

#define DELEGATE_OSTREAM(...)                                                  \
  template <>                                                                  \
  struct delegate_ostream_operator<__VA_ARGS__> : std::true_type {}

namespace FlexFlow {

template <typename T, typename Enable = void>
struct delegate_ostream_operator : std::false_type {};

template <typename T>
typename std::enable_if<delegate_ostream_operator<std::decay_t<T>>::value,
                        std::ostream &>::type
    operator<<(std::ostream &s, T);

} // namespace FlexFlow

namespace fmt {

template <typename T, typename Char>
struct formatter<
  ::std::unordered_set<T>, 
  Char,
  std::enable_if_t<!detail::has_format_as<std::unordered_set<T>>::value>
> : formatter<::std::string, Char> {
  template <typename FormatContext>
  auto format(::std::unordered_set<T> const &m, FormatContext &ctx)
      -> decltype(ctx.out());
};

/* template <typename T> */
/* std::string format_as(::std::unordered_set<T> const &); */

template <typename T, typename Char>
struct formatter<
  ::std::vector<T>,
  Char,
  std::enable_if_t<!detail::has_format_as<std::vector<T>>::value>
> : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::vector<T> const &m, FormatContext &ctx)
      -> decltype(ctx.out());
};

template <typename... Ts>
struct formatter<::std::variant<Ts...>> : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::variant<Ts...> const &m, FormatContext &ctx)
      -> decltype(ctx.out());
};

/* template <typename L, typename R, typename Char> */
/* struct formatter< */
/*   ::std::pair<L, R> */
/*   Char, */
/*   std::enable_if_t<!detail::has_format_as<::std::pair<L, R>>::value> */
/* > : formatter<::std::string> { */
/*   template <typename FormatContext> */
/*   auto format(::std::pair<L, R> const &m, FormatContext &ctx) */
/*       -> decltype(ctx.out()); */

} // namespace fmt

#endif

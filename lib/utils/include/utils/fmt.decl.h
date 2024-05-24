#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H

#include "fmt/format.h"
#include <unordered_set>
#include <vector>

#define CHECK_FMTABLE(...)                                                     \
  static_assert(::FlexFlow::is_fmtable<__VA_ARGS__>::value,                    \
                #__VA_ARGS__ " must be fmtable");

#define DELEGATE_OSTREAM(...)                                                  \
  template <>                                                                  \
  struct delegate_ostream_operator<__VA_ARGS__> : std::true_type {}

namespace FlexFlow {

template <typename T>
using is_fmtable = ::fmt::is_formattable<T>;

template <typename T, typename Enable = void>
struct delegate_ostream_operator : std::false_type {};

template <typename T>
typename std::enable_if<delegate_ostream_operator<std::decay_t<T>>::value,
                        std::ostream &>::type
    operator<<(std::ostream &s, T);

} // namespace FlexFlow

namespace fmt {

template <typename T>
struct formatter<::std::unordered_set<T>> : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::unordered_set<T> const &m, FormatContext &ctx)
      -> decltype(ctx.out());
};

template <typename T>
struct formatter<::std::vector<T>> : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::vector<T> const &m, FormatContext &ctx)
      -> decltype(ctx.out());
};

} // namespace fmt

#endif

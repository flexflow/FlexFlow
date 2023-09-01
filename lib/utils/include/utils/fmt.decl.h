#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H

#include "fmt/format.h"
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename T>
using is_fmtable = ::fmt::is_formattable<T>;

template <typename T, typename Enable = void>
struct already_has_ostream_operator;

template <typename T>
typename std::enable_if<!already_has_ostream_operator<T>::value,
                        std::ostream &>::type
    operator<<(std::ostream &s, T const &t);

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

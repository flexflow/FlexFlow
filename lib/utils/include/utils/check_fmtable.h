#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CHECK_FMTABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CHECK_FMTABLE_H

#include <fmt/format.h>

#define CHECK_FMTABLE(...)                                                     \
  static_assert(::FlexFlow::is_fmtable<__VA_ARGS__>::value,                    \
                #__VA_ARGS__ " must be fmtable");

namespace FlexFlow {

template <typename T>
using is_fmtable = ::fmt::is_formattable<T>;

} // namespace FlexFlow

#endif

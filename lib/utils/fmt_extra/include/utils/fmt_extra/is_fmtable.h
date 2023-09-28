#ifndef _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_IS_FMTABLE_H
#define _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_IS_FMTABLE_H

#include "fmt/format.h"

namespace FlexFlow {

template <typename T>
using is_fmtable = ::fmt::is_formattable<T>;

template <typename T>
inline constexpr bool is_fmtable_v = is_fmtable<T>::value;

#define CHECK_FMTABLE(...)                                                     \
  static_assert(::FlexFlow::is_fmtable<__VA_ARGS__>::value,                    \
                #__VA_ARGS__ " must be fmtable");

}

#endif

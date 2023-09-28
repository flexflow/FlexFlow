#ifndef _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_ALL_ARE_FMTABLE_H
#define _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_ALL_ARE_FMTABLE_H

#include <type_traits>
#include "is_fmtable.h"

namespace FlexFlow {

template <typename... Ts>
using all_are_fmtable = std::conjunction<is_fmtable<Ts>...>;

template <typename... Ts>
inline constexpr bool all_are_fmtable_v = all_are_fmtable<Ts...>::value;

}

#endif

#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "fmt/core.h"
#include "fmt/format.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {

template <typename T, typename Enable = void> 
struct is_fmtable : std::false_type { };

template <typename T>
struct is_fmtable<T, void_t<decltype(fmt::to_string(std::declval<T>()))>>
  : std::true_type { };

}

#endif

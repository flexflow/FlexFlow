#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H

#include "fmt/format.h"
#include "utils/check_fmtable.h"
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

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

#endif

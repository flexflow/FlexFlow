#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_OVERLOAD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_OVERLOAD_H

namespace FlexFlow {

template <class... Ts>
struct overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

} // namespace FlexFlow

#endif

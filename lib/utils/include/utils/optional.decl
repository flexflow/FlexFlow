#ifndef _FLEXFLOW_UTILS_OPTIONAL_H
#define _FLEXFLOW_UTILS_OPTIONAL_H

#include "tl/optional.hpp"

namespace FlexFlow {

using namespace tl;

template <typename T, typename F>
T const &unwrap(optional<T> const &o, F const &f);

template <typename T>
T const &assert_unwrap(optional<T> const &o);

} // namespace FlexFlow

#endif

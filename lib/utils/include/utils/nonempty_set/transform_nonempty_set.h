#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONEMPTY_SET_TRANSFORM_NONEMPTY_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONEMPTY_SET_TRANSFORM_NONEMPTY_SET_H

#include "utils/containers/transform.h"
#include "utils/nonempty_set/nonempty_set.h"

namespace FlexFlow {

template <typename F, typename In, typename Out = std::invoke_result_t<F, In>>
nonempty_set<Out> transform_nonempty_set(nonempty_set<In> const &s,
                                         F const &f) {
  return nonempty_set<Out>{
      transform(s.unwrap_as_set(), f),
  };
}

} // namespace FlexFlow

#endif

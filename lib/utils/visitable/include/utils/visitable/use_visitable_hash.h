#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_USE_VISITABLE_HASH_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_USE_VISITABLE_HASH_H

#include "utils/visitable/visit_hash.h"

namespace FlexFlow {

template <typename T>
struct use_visitable_hash {
  std::size_t operator()(T const &t) const {
    return visit_hash(t);
  }
};

} // namespace FlexFlow

#endif

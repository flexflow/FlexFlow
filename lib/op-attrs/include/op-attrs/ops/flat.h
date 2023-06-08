#ifndef _FLEXFLOW_FLAT_ATTRS_H
#define _FLEXFLOW_FLAT_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct FlatAttrs : public use_visitable_cmp<FlatAttrs> {
  FlatAttrs() = default;
};

} // namespace FlexFlow

VISITABLE_STRUCT_EMPTY(::FlexFlow::FlatAttrs);
MAKE_VISIT_HASHABLE(::FlexFlow::FlatAttrs);

#endif

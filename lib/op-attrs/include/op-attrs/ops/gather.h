#ifndef _FLEXFLOW_GATHER_ATTRS_H
#define _FLEXFLOW_GATHER_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct GatherAttrs : public use_visitable_cmp<GatherAttrs> {
public:
  GatherAttrs() = delete;
  GatherAttrs(ff_dim_t);

public:
  ff_dim_t dim;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::GatherAttrs, dim);
MAKE_VISIT_HASHABLE(::FlexFlow::GatherAttrs);

#endif

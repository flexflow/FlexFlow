#ifndef _FLEXFLOW_REPLICATE_ATTRS_H
#define _FLEXFLOW_REPLICATE_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReplicateAttrs : public use_visitable_cmp<ReplicateAttrs> {
public:
  ReplicateAttrs() = delete;
  ReplicateAttrs(ff_dim_t dim, int degree);

public:
  ff_dim_t replicate_dim;
  int replicate_degree;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ReplicateAttrs, replicate_dim, replicate_degree);
MAKE_VISIT_HASHABLE(::FlexFlow::ReplicateAttrs);

#endif

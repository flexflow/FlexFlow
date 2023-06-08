#ifndef _FLEXFLOW_PARTITION_ATTRS_H
#define _FLEXFLOW_PARTITION_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct RepartitionAttrs : public use_visitable_cmp<RepartitionAttrs> {
public:
  RepartitionAttrs() = delete;
  RepartitionAttrs(ff_dim_t repartition_dim, int repartition_degree);

public:
  ff_dim_t repartition_dim;
  int repartition_degree;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::RepartitionAttrs,
                 repartition_dim,
                 repartition_degree);
MAKE_VISIT_HASHABLE(::FlexFlow::RepartitionAttrs);

#endif

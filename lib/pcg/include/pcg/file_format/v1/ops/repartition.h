#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REPARTITION_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REPARTITION_ATTRS_H

#include "op-attrs/ops/repartition.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1RepartitionAttrs {
  int repartition_dim;
  req<int> repartition_degree;
};
FF_VISITABLE_STRUCT(V1RepartitionAttrs, repartition_dim, repartition_degree);
CHECK_IS_JSONABLE(V1RepartitionAttrs);

V1RepartitionAttrs to_v1(RepartitionAttrs const &a);
RepartitionAttrs from_v1(V1RepartitionAttrs const &va);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REPLICATE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REPLICATE_ATTRS_H

#include "op-attrs/ops/replicate.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReplicateAttrs {
  int replicate_dim;
  req<int> replicate_degree;
};
FF_VISITABLE_STRUCT(V1ReplicateAttrs, replicate_dim, replicate_degree);
CHECK_IS_JSONABLE(V1ReplicateAttrs);

V1ReplicateAttrs to_v1(ReplicateAttrs const &a);
ReplicateAttrs from_v1(V1ReplicateAttrs const &va);

} // namespace FlexFlow

#endif

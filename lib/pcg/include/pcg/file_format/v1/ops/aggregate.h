#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_AGGREGATE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_AGGREGATE_H

#include "op-attrs/ops/aggregate.h"
#include "utils/visitable.h"
#include "utils/json.h"

namespace FlexFlow {

struct V1AggregateAttrs {
  req<int> n;
  req<float> lambda_bal;
};
FF_VISITABLE_STRUCT(V1AggregateAttrs, n, lambda_bal);
CHECK_IS_JSONABLE(V1AggregateAttrs);

V1AggregateAttrs to_v1(AggregateAttrs const &attrs);

} // namespace FlexFlow

#endif

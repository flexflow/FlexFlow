#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_AGGREGATE_SPEC_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_AGGREGATE_SPEC_ATTRS_H

#include "op-attrs/ops/aggregate_spec.h"
#include "utils/json.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1AggregateSpecAttrs {
  req<int> n;
  req<float> lambda_bal;
};
FF_VISITABLE_STRUCT(V1AggregateSpecAttrs, n, lambda_bal);
CHECK_IS_JSONABLE(V1AggregateSpecAttrs);

V1AggregateSpecAttrs to_v1(AggregateSpecAttrs const &a);
AggregateSpecAttrs from_v1(V1AggregateSpecAttrs const &va);

} // namespace FlexFlow

#endif

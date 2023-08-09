#include "op-attrs/ops/aggregate_spec.h"

namespace FlexFlow {

AggregateSpecAttrs::AggregateSpecAttrs(int _n, float _lambda_bal)
    : n(_n), lambda_bal(_lambda_bal) {}

} // namespace FlexFlow

#include "pcg/file_format/v1/ops/aggregate_spec.h"

namespace FlexFlow {

V1AggregateSpecAttrs to_v1(AggregateSpecAttrs const &a) {
  return {a.n, a.lambda_bal};
}

AggregateSpecAttrs from_v1(V1AggregateSpecAttrs const &va) {
  return {va.n, va.lambda_bal};
}

} // namespace FlexFlow

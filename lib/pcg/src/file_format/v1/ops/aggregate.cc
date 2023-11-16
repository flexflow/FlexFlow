#include "pcg/file_format/v1/ops/aggregate.h"

namespace FlexFlow {

V1AggregateAttrs to_v1(AggregateAttrs const &a) {
  return {a.n, a.lambda_bal};
}

AggregateAttrs from_v1(V1AggregateAttrs const &va) {
  return {va.n, va.lambda_bal};
}

} // namespace FlexFlow

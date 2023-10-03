#include "pcg/file_format/v1/ops/aggregate_spec.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1AggregateSpecAttrs to_v1(AggregateSpecAttrs const &a) {
  return {to_v1(a.n), to_v1(a.lambda_bal)};
}

} // namespace FlexFlow

#include "pcg/file_format/v1/ops/aggregate_spec.h"

namespace FlexFlow {

V1AggregateSpecAttrs to_v1(AggregateSpecAttrs const &a) {
  return {a.n, a.lambda_bal};
}

} // namespace FlexFlow

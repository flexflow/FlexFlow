#include "pcg/file_format/v1/ops/aggregate.h"

namespace FlexFlow {

V1AggregateAttrs to_v1(AggregateAttrs const &a) {
  return {a.n, a.lambda_bal};
}

} // namespace FlexFlow

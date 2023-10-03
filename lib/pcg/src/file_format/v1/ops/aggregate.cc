#include "pcg/file_format/v1/ops/aggregate.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1AggregateAttrs to_v1(AggregateAttrs const &a) {
  return {to_v1(a.n), to_v1(a.lambda_bal)};
}

} // namespace FlexFlow

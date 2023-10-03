#include "pcg/file_format/v1/ops/reduction.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ReductionAttrs to_v1(ReductionAttrs const &a) {
  return {to_v1(a.reduction_dim), to_v1(a.reduction_degree)};
}

} // namespace FlexFlow

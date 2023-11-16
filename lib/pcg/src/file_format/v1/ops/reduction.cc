#include "pcg/file_format/v1/ops/reduction.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1ReductionAttrs to_v1(ReductionAttrs const &a) {
  return {to_v1(a.reduction_dim), a.reduction_degree};
}

ReductionAttrs from_v1(V1ReductionAttrs const &va) {
  return {ff_dim_t(va.reduction_dim), va.reduction_degree};
}

} // namespace FlexFlow

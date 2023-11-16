#include "pcg/file_format/v1/ops/softmax.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1SoftmaxAttrs to_v1(SoftmaxAttrs const &a) {
  return {to_v1(a.dim)};
}

SoftmaxAttrs from_v1(V1SoftmaxAttrs const &va) {
  return {ff_dim_t(va.dim)};
}

} // namespace FlexFlow

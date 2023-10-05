#include "pcg/file_format/v1/ops/gather.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1GatherAttrs to_v1(GatherAttrs const &a) {
  return {to_v1(a.dim)};
}

GatherAttrs from_v1(V1GatherAttrs const &va) {
  return {ff_dim_t(va.dim)};
}

} // namespace FlexFlow

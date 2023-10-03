#include "pcg/file_format/v1/ops/gather.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1GatherAttrs to_v1(GatherAttrs const &a) {
  return {to_v1(a.dim)};
}

} // namespace FlexFlow

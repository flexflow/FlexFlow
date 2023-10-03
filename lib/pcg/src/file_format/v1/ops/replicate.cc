#include "pcg/file_format/v1/ops/replicate.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ReplicateAttrs to_v1(ReplicateAttrs const &a) {
  return {to_v1(a.replicate_dim), to_v1(a.replicate_degree)};
}

} // namespace FlexFlow

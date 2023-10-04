#include "pcg/file_format/v1/ops/replicate.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1ReplicateAttrs to_v1(ReplicateAttrs const &a) {
  return {to_v1(a.replicate_dim), a.replicate_degree};
}

} // namespace FlexFlow

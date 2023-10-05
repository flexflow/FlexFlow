#include "pcg/file_format/v1/ops/replicate.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1ReplicateAttrs to_v1(ReplicateAttrs const &a) {
  return {to_v1(a.replicate_dim), a.replicate_degree};
}

ReplicateAttrs from_v1(V1ReplicateAttrs const &va) {
  return {ff_dim_t(va.replicate_dim), va.replicate_degree};
}

} // namespace FlexFlow

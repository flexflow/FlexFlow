#include "pcg/file_format/v1/ops/element_binary.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ElementBinaryAttrs to_v1(ElementBinaryAttrs const &a) {
  return {to_v1(a.type),
          to_v1(a.compute_type),
          to_v1(a.should_broadcast_lhs),
          to_v1(a.should_broadcast_rhs)};
}

} // namespace FlexFlow

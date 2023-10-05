#include "pcg/file_format/v1/ops/element_binary.h"

namespace FlexFlow {

V1ElementBinaryAttrs to_v1(ElementBinaryAttrs const &a) {
  return {to_v1(a.type),
          to_v1(a.compute_type),
          a.should_broadcast_lhs,
          a.should_broadcast_rhs};
}

ElementBinaryAttrs from_v1(V1ElementBinaryAttrs const &va) {
  return {from_v1(va.type),
          from_v1(va.compute_type),
          va.should_broadcast_lhs,
          va.should_broadcast_rhs};
}

} // namespace FlexFlow

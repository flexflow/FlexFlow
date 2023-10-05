#include "pcg/file_format/v1/ops/reverse.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1ReverseAttrs to_v1(ReverseAttrs const &a) {
  return {to_v1(a.axis)};
}

ReverseAttrs from_v1(V1ReverseAttrs const &va) {
  return {from_v1(va.axis)};
}

} // namespace FlexFlow

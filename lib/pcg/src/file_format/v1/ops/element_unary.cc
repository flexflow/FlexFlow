#include "pcg/file_format/v1/ops/element_unary.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ElementScalarUnaryAttrs to_v1(ElementScalarUnaryAttrs const &a) {
  return {to_v1(a.op), to_v1(a.scalar)};
}

V1ElementUnaryAttrs to_v1(ElementUnaryAttrs const &a) {
  return {to_v1(a.op)};
}

} // namespace FlexFlow

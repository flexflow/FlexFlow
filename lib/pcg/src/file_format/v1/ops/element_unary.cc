#include "pcg/file_format/v1/ops/element_unary.h"

namespace FlexFlow {

V1ElementScalarUnaryAttrs to_v1(ElementScalarUnaryAttrs const &a) {
  return {to_v1(a.op), a.scalar};
}

ElementScalarUnaryAttrs from_v1(V1ElementScalarUnaryAttrs const &va) {
  return {from_v1(va.op), va.scalar};
}

V1ElementUnaryAttrs to_v1(ElementUnaryAttrs const &a) {
  return {to_v1(a.op)};
}

ElementUnaryAttrs from_v1(V1ElementUnaryAttrs const &va) {
  return {from_v1(va.op)};
}

} // namespace FlexFlow

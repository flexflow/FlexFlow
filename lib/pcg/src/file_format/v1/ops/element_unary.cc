#include "pcg/file_format/v1/ops/element_unary.h"

namespace FlexFlow {

V1ElementScalarUnaryAttrs to_v1(ElementScalarUnaryAttrs const &a) {
  return {to_v1(a.op), a.scalar};
}

V1ElementUnaryAttrs to_v1(ElementUnaryAttrs const &a) {
  return {to_v1(a.op)};
}

} // namespace FlexFlow

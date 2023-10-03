#include "pcg/file_format/v1/ops/reshape.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ReshapeAttrs to_v1(ReshapeAttrs const &a) {
  return {to_v1(a.shape)};
}

} // namespace FlexFlow

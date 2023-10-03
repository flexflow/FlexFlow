#include "pcg/file_format/v1/ops/batch_norm.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1BatchNormAttrs to_v1(BatchNormAttrs const &a) {
  return {to_v1(a.relu)};
}

} // namespace FlexFlow

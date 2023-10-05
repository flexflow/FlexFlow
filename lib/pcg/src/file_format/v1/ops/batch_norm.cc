#include "pcg/file_format/v1/ops/batch_norm.h"

namespace FlexFlow {

V1BatchNormAttrs to_v1(BatchNormAttrs const &a) {
  return {a.relu};
}

BatchNormAttrs from_v1(V1BatchNormAttrs const &va) {
  return {va.relu};
}

} // namespace FlexFlow

#include "pcg/file_format/v1/ops/batch_norm.h"

namespace FlexFlow {

V1BatchNormAttrs to_v1(BatchNormAttrs const &a) {
  return {a.relu};
}

} // namespace FlexFlow

#include "pcg/file_format/v1/ops/dropout.h"

namespace FlexFlow {

V1DropoutAttrs to_v1(DropoutAttrs const &a) {
  return {a.rate, a.seed};
}

} // namespace FlexFlow

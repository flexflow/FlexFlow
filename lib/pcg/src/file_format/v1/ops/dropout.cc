#include "pcg/file_format/v1/ops/dropout.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1DropoutAttrs to_v1(DropoutAttrs const &a) {
  return {to_v1(a.rate), to_v1(a.seed)};
}

} // namespace FlexFlow

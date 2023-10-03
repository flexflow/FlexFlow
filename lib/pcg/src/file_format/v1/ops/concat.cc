#include "pcg/file_format/v1/ops/concat.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ConcatAttrs to_v1(ConcatAttrs const &a) {
  return {to_v1(a.axis)};
}

} // namespace FlexFlow

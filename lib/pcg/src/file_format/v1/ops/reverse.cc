#include "pcg/file_format/v1/ops/reverse.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1ReverseAttrs to_v1(ReverseAttrs const &a) {
  return {to_v1(a.axis)};
}

} // namespace FlexFlow

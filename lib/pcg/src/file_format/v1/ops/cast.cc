#include "pcg/file_format/v1/ops/cast.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1CastAttrs to_v1(CastAttrs const &a) {
  return {to_v1(a.dtype)};
}

} // namespace FlexFlow

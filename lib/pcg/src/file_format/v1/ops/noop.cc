#include "pcg/file_format/v1/ops/noop.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1NoopAttrs to_v1(NoopAttrs const &a) {
  return {
      // No fields in NoopAttrs.
  };
}

NoopAttrs from_v1(V1NoopAttrs const &va) {
  return {
      // No fields in NoopAttrs.
  };
}

} // namespace FlexFlow

#include "pcg/file_format/v1/ops/groupby.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1Group_byAttrs to_v1(Group_byAttrs const &a) {
  return {to_v1(a.n), to_v1(a.alpha)};
}

} // namespace FlexFlow

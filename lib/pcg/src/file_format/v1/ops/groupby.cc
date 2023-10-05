#include "pcg/file_format/v1/ops/groupby.h"

namespace FlexFlow {

V1Group_byAttrs to_v1(Group_byAttrs const &a) {
  return {a.n, a.alpha};
}

Group_byAttrs from_v1(V1Group_byAttrs const &va) {
  return {va.n, va.alpha};
}

} // namespace FlexFlow

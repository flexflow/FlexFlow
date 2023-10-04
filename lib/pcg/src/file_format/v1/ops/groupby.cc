#include "pcg/file_format/v1/ops/groupby.h"

namespace FlexFlow {

V1Group_byAttrs to_v1(Group_byAttrs const &a) {
  return {a.n, a.alpha};
}

} // namespace FlexFlow

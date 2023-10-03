#include "pcg/file_format/v1/ops/split.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1SplitAttrs to_v1(SplitAttrs const &a) {
  return {std::vector<int>(a.splits.begin(), a.splits.end()), to_v1(a.axis)};
}

} // namespace FlexFlow

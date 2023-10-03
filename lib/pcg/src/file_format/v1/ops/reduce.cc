#include "pcg/file_format/v1/ops/reduce.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ReduceAttrs to_v1(ReduceAttrs const &a) {
  return {std::vector<int>(a.axes.begin(), a.axes.end()),
          to_v1(a.op_type),
          to_v1(a.keepdims)};
}

} // namespace FlexFlow

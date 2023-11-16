#include "pcg/file_format/v1/ops/reduce.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1ReduceAttrs to_v1(ReduceAttrs const &a) {
  return {std::vector<int>(a.axes.begin(), a.axes.end()),
          to_v1(a.op_type),
          a.keepdims};
}

ReduceAttrs from_v1(V1ReduceAttrs const &va) {
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> axes;
  for (int const &d : va.axes) {
    axes.push_back(ff_dim_t(d));
  }
  return {axes, from_v1(va.op_type), va.keepdims};
}

} // namespace FlexFlow

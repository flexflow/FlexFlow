#include "op-attrs/ops/cast.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(CastAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

} // namespace FlexFlow

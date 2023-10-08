#include "op-attrs/ops/cast.h"

namespace FlexFlow {

bool CastAttrs::is_valid(ParallelTensorShape const &input) const {
    if (!input.is_valid()) {
        return false;
    }
    return true;
}

ParallelTensorShape get_output_shape(CastAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

} // namespace FlexFlow

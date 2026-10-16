#include "op-attrs/ops/cast.h"
#include "op-attrs/datatype.h"

namespace FlexFlow {

TensorShape cast_get_output_shape(CastAttrs const &attrs,
                                  TensorShape const &input) {

  TensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

ParallelTensorShape
    cast_get_output_parallel_shape(CastAttrs const &attrs,
                                   ParallelTensorShape const &input) {

  ParallelTensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

} // namespace FlexFlow

#include "op-attrs/ops/reshape.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

// pytorch: the input: [2,3,4], shape maybe [-1,6]， should we add this? and the
// output is [4, 6]
bool ReshapeAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  std::size_t input_volume = 1;
  for (int i = 0; i < input.num_dims(); i++) {
    input_volume *= input.at(ff_dim_t(i)).size;
  }
  std::size_t attrs_volume = 1;
  for (int i = 0; i < this->shape.dims.num_dims(); i++) {
    attrs_volume *= this->shape.at(ff_dim_t(i));
  }
  return (input_volume == attrs_volume);
}

// pytorch: the input: [2,3,4], shape maybe [-1,6]， should we add this? and the
// output is [4, 6] currently we doesn't consider the case of -1,we can support
// this later the input:[2,3,4], attrs.shape:[4,6], the output is [4, 6]
ParallelTensorShape get_output_shape(ReshapeAttrs const &attrs,
                                     ParallelTensorShape const &input) {

  assert(attrs.is_valid(input) && "input is not valid");
  ParallelTensorDims dims{attrs.shape.dims};
  ParallelTensorShape output{dims, input.data_type};
  return output;
}

} // namespace FlexFlow

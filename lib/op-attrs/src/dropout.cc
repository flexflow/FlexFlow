#include "dropout.h"
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {    

bool DropoutAttrs::is_valid(ParallelTensorShape const & input) const {
    if(!input.is_valid()) {
        return false;
    }
    return true;
}

ParallelTensorShape get_output_shape(DropoutAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
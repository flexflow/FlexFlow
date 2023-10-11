#include "op-attrs/ops/reverse.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

bool ReverseAttrs::is_valid(ParallelTensorShape const & input) const {
    if(input.is_valid() ==false) {
        return false;
    }
    if(this->axis < 0 || this->axis >= input.num_dims()) {
        return false;
    }
    return true;
}

ParallelTensorShape get_output_shape(ReverseAttrs const & attrs, 
                                     ParallelTensorShape const & input) {
    ParallelTensorShape output = input;
    return output;
}


};
#include "op-attrs/ops/linear.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

bool LinearAttrs::is_valid(ParallelTensorShape const & input) const {
    if(!input.is_valid()) {
        return false;
    }
    return true;
}

//pytorch: input shape:{batch_size, input_channels}
//pytorch linearattrs: should be {input_channels, output_channels} 
//pytorch: output shape:{batch_size, output_channels}
//question: the Linearattrs doesn't have input_channels
ParallelTensorShape get_output_shape(LinearAttrs const & atts,
                                     ParallelTensorShape const & input) {
    ParallelTensorShape out_shape = input;
    out_shape.at(ff_dim_t(0)).size = atts.out_channels;
    return out_shape;
}

} // namespace FlexFlow

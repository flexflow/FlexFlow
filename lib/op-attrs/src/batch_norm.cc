#include "op-attrs/ops/batch_norm.h"

namespace FlexFlow {
 
 bool BatchNormAttrs::is_valid(ParallelTensorShape const & input) {
     if(!input.is_valid()) {
         return false;
     }
    return true;
 }

ParallelTensorShape get_output_shape(BatchNormAttrs const & attrs, ParallelTensorShape const & input) {
    return input; 
}

} // namespace FlexFlow

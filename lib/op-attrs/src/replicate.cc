#include "op-attrs/ops/replicate.h"
#include "utils/exception.decl.h"

namespace FlexFlow {

bool ReplicateAttrs::is_valid(ParallelTensorShape const &input) const {
  NOT_IMPLEMENTED();
}

//replicate by n multiplies degree by n and shape by n
ParallelTensorShape get_output_shape(ReplicateAttrs const & attrs,
                                     ParallelTensorShape const & input) {
  NOT_IMPLEMENTED();
}



} // namespace FlexFlow

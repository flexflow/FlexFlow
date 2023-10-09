#include "op-attrs/ops/reduce.h"
#include "utils/exception.decl.h"

namespace FlexFlow {

bool ReduceAttrs::is_valid(ParallelTensorShape const & input) const {
    NOT_IMPLEMENTED()
}

ParallelTensorShape get_output_shape(ReduceAttrs const & attrs,
                                     ParallelTensorShape const & input) {
    NOT_IMPLEMENTED()
}

} // namespace FlexFlow

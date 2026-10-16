#include "op-attrs/ops/reduce.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

ParallelTensorShape
    reduce_get_output_parallel_shape(ReduceAttrs const &,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

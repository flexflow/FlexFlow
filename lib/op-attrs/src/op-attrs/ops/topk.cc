#include "op-attrs/ops/topk.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape topk_get_output_shape(TopKAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape
    topk_get_output_parallel_shape(TopKAttrs const &attrs,
                                   ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

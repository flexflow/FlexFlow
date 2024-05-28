#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_MULTIHEAD_ATTENTION_PARALLEL_INPUTS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_MULTIHEAD_ATTENTION_PARALLEL_INPUTS_H

#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_attention_parallel_input_shape(ParallelTensorShape const &input_q,
                                                                                         ParallelTensorShape const &input_k,
                                                                                         ParallelTensorShape const &input_v);

} // namespace FlexFlow

#endif

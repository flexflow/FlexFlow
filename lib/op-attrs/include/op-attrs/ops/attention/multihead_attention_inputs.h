#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_MULTIHEAD_ATTENTION_INPUTS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_MULTIHEAD_ATTENTION_INPUTS_H

#include "op-attrs/ops/attention/multihead_attention_inputs.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

tl::expected<MultiHeadAttentionInputs, std::string> parse_attention_input_shape(TensorShape const &input_q,
                                                     TensorShape const &input_k,
                                                     TensorShape const &input_v);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/activation.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct L1RegularizerAttrs {
  req<float> lambda;
};
FF_VISITABLE_STRUCT(L1RegularizerAttrs, lambda);
CHECK_VALID_OP_ATTR(L1RegularizerAttrs);

struct L2RegularizerAttrs {
  req<float> lambda;
};
FF_VISITABLE_STRUCT(L2RegularizerAttrs, lambda);
CHECK_VALID_OP_ATTR(L2RegularizerAttrs);

using RegularizerAttrs = std::variant<L1RegularizerAttrs, L2RegularizerAttrs>;

struct LinearAttrs {
  int out_channels;
  bool use_bias;
  DataType data_type;
  std::optional<Activation> activation;
  req<std::optional<RegularizerAttrs>> regularizer;
};
FF_VISITABLE_STRUCT(
    LinearAttrs, out_channels, use_bias, data_type, activation, regularizer);
CHECK_VALID_OP_ATTR(LinearAttrs);

TensorShape get_weights_shape(LinearAttrs const &attrs,
                              TensorShape const &input);
ParallelTensorShape get_weights_shape(LinearAttrs const &attrs,
                                      ParallelTensorShape const &input);
TensorShape get_bias_shape(LinearAttrs const &attrs, TensorShape const &input);
ParallelTensorShape get_bias_shape(LinearAttrs const &attrs,
                                   ParallelTensorShape const &input);

} // namespace FlexFlow

#endif

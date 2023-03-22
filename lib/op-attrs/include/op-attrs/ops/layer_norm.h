#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct LayerNormAttrs : public UnaryOpAttrs {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
};

bool operator==(LayerNormAttrs const &, LayerNormAttrs const &);
bool operator<(LayerNormAttrs const &, LayerNormAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::LayerNormAttrs, axes, elementwise_affine, eps);

namespace std {
template <>
struct hash<::FlexFlow::LayerNormAttrs> {
  size_t operator()(::FlexFlow::LayerNormAttrs const &) const;
};
}

#endif 

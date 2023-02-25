#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct LayerNormParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
};

bool operator==(LayerNormParams const &, LayerNormParams const &);
bool operator<(LayerNormParams const &, LayerNormParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::LayerNormParams, axes, elementwise_affine, eps);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::LayerNormParams> {
  size_t operator()(::FlexFlow::opmeta::LayerNormParams const &) const;
};
}

#endif // _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H

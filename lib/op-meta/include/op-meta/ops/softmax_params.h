#ifndef _FLEXFLOW_SOFTMAX_PARAMS_H
#define _FLEXFLOW_SOFTMAX_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct SoftmaxParams : public UnaryOpParams {
public:
  OperatorType op_type() const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
public:
  int dim;
};
bool operator==(SoftmaxParams const &, SoftmaxParams const &);
bool operator<(SoftmaxParams const &, SoftmaxParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::SoftmaxParams, dim);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::SoftmaxParams> {
  size_t operator()(::FlexFlow::opmeta::SoftmaxParams const &) const;
};
}

#endif // _FLEXFLOW_SOFTMAX_PARAMS_H

#ifndef _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct ElementUnaryParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  OperatorType op;
  /* bool inplace; */
  float scalar = 0.0;
};

bool operator==(ElementUnaryParams const &, ElementUnaryParams const &);
bool operator<(ElementUnaryParams const &, ElementUnaryParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::ElementUnaryParams, op, scalar);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::ElementUnaryParams> {
  size_t operator()(::FlexFlow::opmeta::ElementUnaryParams const &) const;
};
} 

#endif // _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H

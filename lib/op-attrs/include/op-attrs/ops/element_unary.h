#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementUnaryAttrs {
/* public: */
  /* ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override; */
  /* OperatorType op_type() const override; */
public:
  OperatorType op;
  /* bool inplace; */
  float scalar = 0.0;
};

bool operator==(ElementUnaryAttrs const &, ElementUnaryAttrs const &);
bool operator<(ElementUnaryAttrs const &, ElementUnaryAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ElementUnaryAttrs, op, scalar);

namespace std {
template <>
struct hash<::FlexFlow::ElementUnaryAttrs> {
  size_t operator()(::FlexFlow::ElementUnaryAttrs const &) const;
};
} 

#endif 

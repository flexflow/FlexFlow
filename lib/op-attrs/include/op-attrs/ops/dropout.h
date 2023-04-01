#ifndef _FLEXFLOW_DROPOUT_ATTRS_H
#define _FLEXFLOW_DROPOUT_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct DropoutAttrs : public UnaryOpAttrs {
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  float rate;
  unsigned long long seed;
};

bool operator==(DropoutAttrs const &, DropoutAttrs const &);
bool operator<(DropoutAttrs const &, DropoutAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::DropoutAttrs, rate, seed);

namespace std {
template <>
struct hash<::FlexFlow::DropoutAttrs> {
  size_t operator()(::FlexFlow::DropoutAttrs const &) const;
};
} 

#endif 

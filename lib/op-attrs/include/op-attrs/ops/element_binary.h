#ifndef _FLEXFLOW_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_ELEMENT_BINARY_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/binary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementBinaryAttrs : public BinaryOpAttrs {
public:
  bool is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
public:
  OperatorType type;
};

bool operator==(ElementBinaryAttrs const &, ElementBinaryAttrs const &);
bool operator<(ElementBinaryAttrs const &, ElementBinaryAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ElementBinaryAttrs, type);

namespace std {
template <>
struct hash<::FlexFlow::ElementBinaryAttrs> {
  size_t operator()(::FlexFlow::ElementBinaryAttrs const &) const;
};
}

#endif 

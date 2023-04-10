#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementScalarUnaryAttrs {
public:
  ElementScalarUnaryAttrs(OperatorType, float);
public:
  OperatorType op;
  /* bool inplace; */
  float scalar;
};

bool operator==(ElementScalarUnaryAttrs const &, ElementScalarUnaryAttrs const &);
bool operator<(ElementScalarUnaryAttrs const &, ElementScalarUnaryAttrs const &);

struct ElementUnaryAttrs {
public:
  ElementUnaryAttrs(OperatorType);
public:
  OperatorType op;
};

bool operator==(ElementUnaryAttrs const &, ElementUnaryAttrs const &);
bool operator<(ElementUnaryAttrs const &, ElementUnaryAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ElementScalarUnaryAttrs, op, scalar);
VISITABLE_STRUCT(::FlexFlow::ElementUnaryAttrs, op);

namespace std {
template <>
struct hash<::FlexFlow::ElementScalarUnaryAttrs> {
  size_t operator()(::FlexFlow::ElementScalarUnaryAttrs const &) const;
};

template <>
struct hash<::FlexFlow::ElementUnaryAttrs> {
  size_t operator()(::FlexFlow::ElementUnaryAttrs const &) const;
};
} 

#endif 

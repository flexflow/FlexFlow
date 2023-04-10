#ifndef _FLEXFLOW_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_ELEMENT_BINARY_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/binary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementBinaryAttrs {
public:
  ElementBinaryAttrs(OperatorType, bool should_broadcast_lhs, bool should_broadcast_rhs);
public:
  OperatorType type;
  bool should_broadcast_lhs;
  bool should_broadcast_rhs;
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

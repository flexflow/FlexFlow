#ifndef _FLEXFLOW_CAST_ATTRS_H
#define _FLEXFLOW_CAST_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct CastAttrs : public UnaryOpAttrs {
  bool is_valid(ParallelTensorShape const &) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  DataType dtype;
};

bool operator==(CastAttrs const &, CastAttrs const &);
bool operator<(CastAttrs const &, CastAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::CastAttrs, dtype);

namespace std {
template <>
struct hash<::FlexFlow::CastAttrs> {
  size_t operator()(::FlexFlow::CastAttrs const &) const;
};
} 

#endif 

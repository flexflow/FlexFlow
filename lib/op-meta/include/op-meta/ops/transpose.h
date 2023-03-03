#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct TransposeAttrs : public UnaryOpAttrs {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> perm;
};

bool operator==(TransposeAttrs const &, TransposeAttrs const &);
bool operator<(TransposeAttrs const &, TransposeAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::TransposeAttrs, perm);

namespace std {
template <>
struct hash<::FlexFlow::TransposeAttrs> {
  size_t operator()(::FlexFlow::TransposeAttrs const &) const;
};
} 

#endif 

#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct TransposeAttrs {
/* public: */
/*   ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override; */
/*   OperatorType op_type() const override; */
/* public: */
  stack_vector<int, MAX_TENSOR_DIM> perm;
};

bool operator==(TransposeAttrs const &, TransposeAttrs const &);
bool operator!=(TransposeAttrs const &, TransposeAttrs const &);
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

#ifndef _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "unary_op.h"
#include "utils/visitable.h"
#include "utils/stack_vector.h"

namespace FlexFlow {

struct ReduceAttrs {
/* public: */
/*   ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override; */
/*   OperatorType op_type() const override; */
/*   bool is_valid(ParallelTensorShape const &) const override; */
public:
  stack_vector<int, MAX_TENSOR_DIM> axes;
  OperatorType op_type;
  bool keepdims;
};

bool operator==(ReduceAttrs const &, ReduceAttrs const &);
bool operator!=(ReduceAttrs const &, ReduceAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ReduceAttrs, axes, keepdims);

namespace std {
template <>
struct hash<::FlexFlow::ReduceAttrs> {
  size_t operator()(::FlexFlow::ReduceAttrs const &) const;
};
}

#endif

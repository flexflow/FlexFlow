#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct Pool2DAttrs : public UnaryOpAttrs {
public:
  void solve_dims(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;

  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input) const;

  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
};

bool operator==(Pool2DAttrs const &, Pool2DAttrs const &);
bool operator<(Pool2DAttrs const &, Pool2DAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::Pool2DAttrs, 
                 kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type, activation);

namespace std {
template <>
struct hash<::FlexFlow::Pool2DAttrs> {
  size_t operator()(::FlexFlow::Pool2DAttrs const &) const;
};
} 

#endif 

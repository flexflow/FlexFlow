#ifndef _FLEXFLOW_POOL_2D_PARAMS_H
#define _FLEXFLOW_POOL_2D_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct Pool2DParams : public UnaryOpParams {
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

bool operator==(Pool2DParams const &, Pool2DParams const &);
bool operator<(Pool2DParams const &, Pool2DParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::Pool2DParams, 
                 kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type, activation);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::Pool2DParams> {
  size_t operator()(::FlexFlow::opmeta::Pool2DParams const &) const;
};
} 

#endif // _FLEXFLOW_POOL_2D_PARAMS_H

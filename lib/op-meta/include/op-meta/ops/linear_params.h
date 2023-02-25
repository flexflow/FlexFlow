#ifndef _FLEXFLOW_LINEAR_PARAMS_H
#define _FLEXFLOW_LINEAR_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct LinearParams : public UnaryOpParams {
public:
  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_kernel_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_bias_shape(ParallelTensorShape const &input_shape) const;

  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int out_channels;
  bool use_bias;
  DataType data_type;
  ActiMode activation;
};

bool operator==(LinearParams const &, LinearParams const &);
bool operator<(LinearParams const &, LinearParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::LinearParams, out_channels, use_bias, data_type, activation);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::LinearParams> {
  size_t operator()(::FlexFlow::opmeta::LinearParams const &) const;
};
} 

#endif // _FLEXFLOW_LINEAR_PARAMS_H

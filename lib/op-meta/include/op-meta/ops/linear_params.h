#ifndef _FLEXFLOW_LINEAR_PARAMS_H
#define _FLEXFLOW_LINEAR_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct LinearParams : public OpParamsInterface {
public:
  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_kernel_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_bias_shape(ParallelTensorShape const &input_shape) const;

  using AsConstTuple = std::tuple<int, bool, DataType, ActiMode>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  int out_channels;
  bool use_bias;
  DataType data_type;
  ActiMode activation;
};

bool operator==(LinearParams const &, LinearParams const &);
bool operator<(LinearParams const &, LinearParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::LinearParams> {
  size_t operator()(FlexFlow::LinearParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_LINEAR_PARAMS_H

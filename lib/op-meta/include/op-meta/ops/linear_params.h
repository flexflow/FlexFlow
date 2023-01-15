#ifndef _FLEXFLOW_LINEAR_PARAMS_H
#define _FLEXFLOW_LINEAR_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

class LinearParams {
public:
  int out_channels;
  bool use_bias;
  DataType data_type;
  ActiMode activation;

  bool is_valid(ParallelTensorShape const &input_shape) const;
  void solve_dims(ParallelTensorShape const &input_shape,
                  ParallelTensorShape &output_shape,
                  ParallelTensorShape &kernel_shape,
                  ParallelTensorShape &bias_shape) const;

  enum NamedDimensions {
    INPUT_CHANNEL,
    INPUT_SAMPLE,
    INPUT_REPLICA,
    OUTPUT_CHANNEL,
    OUTPUT_SAMPLE,
    OUTPUT_REPLICA,
    KERNEL_CHANNEL_IN,
    KERNEL_CHANNEL_OUT,
    BIAS_CHANNEL_OUT
  };

  std::unordered_map<NamedDimensions, int>
      get_dimension_names(ParallelTensorShape const &input_name) const;

  friend bool operator==(LinearParams const &lhs, LinearParams const &rhs);

private:
  void mark_replica_dims(ParallelTensorShape const &input_shape,
                         ParallelTensorShape &output_shape,
                         ParallelTensorShape &kernel_shape,
                         ParallelTensorShape &bias_shape) const;
  void calculate_nonreplica_dim_sizes(ParallelTensorShape const &input_shape,
                                      ParallelTensorShape &output_shape,
                                      ParallelTensorShape &kernel_shape,
                                      ParallelTensorShape &bias_shape) const;
};

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::LinearParams> {
  size_t operator()(FlexFlow::LinearParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_LINEAR_PARAMS_H

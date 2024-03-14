#ifndef _FLEXFLOW_LINEAR_PARAMS_H
#define _FLEXFLOW_LINEAR_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

class LinearParams {
public:
  LayerID layer_guid;
  int out_channels;
  bool use_bias;
  DataType data_type;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  DataType quantization_type;
  bool offload;
  char name[MAX_OPNAME];

  bool is_valid(ParallelTensorShape const &input_shape) const;
  void solve_dims(const ParallelTensor input,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM],
                  int *kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM],
                  int *bias_ndims) const;
  void solve_dims(ParallelTensorShape const &input_shape,
                  ParallelTensorShape &output_shape,
                  ParallelTensorShape &kernel_shape,
                  ParallelTensorShape &bias_shape) const;
  void solve_dims(ParallelTensorShape const &input_shape,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM],
                  int *kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM],
                  int *bias_ndims) const;
  void construct_mappings(std::vector<ParallelDimMappingRecord> &,
                          ParallelTensorShape const &) const;

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
                         ParallelDim output_dims[MAX_TENSOR_DIM],
                         ParallelDim kernel_dims[MAX_TENSOR_DIM],
                         ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
  void calculate_nonreplica_dim_sizes(ParallelTensorShape const &input_shape,
                                      ParallelDim output_dims[MAX_TENSOR_DIM],
                                      int *output_ndims,
                                      ParallelDim kernel_dims[MAX_TENSOR_DIM],
                                      int *kernel_ndims,
                                      ParallelDim bias_dims[MAX_TENSOR_DIM],
                                      int *bias_ndims) const;
};

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::LinearParams> {
  size_t operator()(FlexFlow::LinearParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_LINEAR_PARAMS_H

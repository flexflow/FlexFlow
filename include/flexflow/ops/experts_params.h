#pragma once

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ExpertsParams {
  LayerID layer_guid;
  int num_experts;
  int experts_start_idx;
  int experts_output_dim_size;
  float alpha;
  int experts_num_layers;
  int experts_internal_dim_size;
  bool use_bias;
  ActiMode activation;

  bool is_valid(std::vector<ParallelTensorShape> const &) const;
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

  std::unordered_map<NamedDimensions, int>
      get_dimension_names(ParallelTensorShape const &input_name) const;

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

bool operator==(ExpertsParams const &, ExpertsParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ExpertsParams> {
  size_t operator()(FlexFlow::ExpertsParams const &) const;
};
} // namespace std

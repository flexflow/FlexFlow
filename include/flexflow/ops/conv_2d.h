#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/node.h"
#include "flexflow/device.h"

namespace FlexFlow {
  
class FFModel;
class Layer;
  
namespace Conv2DInput {
  static constexpr int INDEX = 0;

  enum {
    WIDTH = 0,
    HEIGHT = 1,
    CHANNEL = 2,
    SAMPLE = 3,
    REPLICA = 4,
    NUMDIM
  };
}

namespace Conv2DOutput {
  enum {
    WIDTH = 0,
    HEIGHT = 1,
    CHANNEL = 2,
    SAMPLE = 3,
    REPLICA = 4,
    NUMDIM
  };
}

namespace Conv2DKernel {
  static constexpr int INDEX = 0;

  enum {
    WIDTH = 0,
    HEIGHT = 1,
    CHANNEL_IN = 2,
    CHANNEL_OUT = 3,
    REPLICA = 4,
    NUMDIM
  };
}

namespace Conv2DBias {
  static constexpr int INDEX = 1;

  enum {
    CHANNEL = 0,
    REPLICA_1 = 1,
    REPLICA_2 = 2,
    REPLICA_3 = 3,
    REPLICA_4 = 4,
    NUMDIM
  };
}

struct Conv2DParams {
  LayerID layer_guid;
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
  ActiMode activation;
  bool use_bias;

  bool is_valid(ParallelTensorShape const &input) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
  // size_t get_hash(const ParallelTensor input) const;

  friend bool operator==(Conv2DParams const &lhs, Conv2DParams const &rhs);
private:
  void mark_replica_dims(ParallelTensorShape const &input, 
                         ParallelDim output_dims[MAX_TENSOR_DIM],
                         ParallelDim kernel_dims[MAX_TENSOR_DIM],
                         ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
  int output_size(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const; 
  int kernel_size(ParallelTensorShape const &input_shape,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM]) const; 
  int bias_size(ParallelTensorShape const &input,
                ParallelDim bias_dims[MAX_TENSOR_DIM]) const; 
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handler);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
#else
  miopenTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  miopenTensorDescriptor_t filterDesc;
  miopenActivationDescriptor_t actiDesc;
  miopenConvolutionDescriptor_t convDesc;
  miopenConvFwdAlgorithm_t fwdAlgo;
  miopenConvBwdWeightsAlgorithm_t bwdFilterAlgo;
  miopenConvBwdDataAlgorithm_t bwdDataAlgo;
#endif
  bool relu, use_bias;
  char op_name[MAX_OPNAME];
};

class Conv2D : public Op {
public:
  using Params = Conv2DParams;

  Conv2D(FFModel& model,
         const LayerID& layer_guid,
         const ParallelTensor input,
         int outChannels,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         ActiMode activation,
         int groups,
         bool use_bias,
         bool allocate_weights,
         const char* name);
  Conv2D(FFModel& model,
         Conv2D const &other, 
         const ParallelTensor input,
         bool allocate_weights);
  Conv2D(FFModel& model,
         Conv2DParams const &params,
         ParallelTensor input,
         bool allocate_weights,
         const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  //void update(const FFModel&);
  void print_layer(const FFModel& model) override;
  //Parameter* get_parameter(int index);
  //void create_weights(FFModel& model);
  //void create_input_partition(FFModel& model);
  static Op* create_operator_from_layer(FFModel& model,
                                        const Layer* layer,
                                        const std::vector<ParallelTensor>& inputs);

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void init_kernel(const Conv2D *conv, 
                          Conv2DMeta *m,
                          int input_w, int input_h, int input_c, int input_n,
                          int output_w, int output_h, int output_c, int output_n,
                          int pad_h, int pad_w,
                          const float* input_ptr, float* output_ptr, const float* kernel_ptr, float* kernel_grad_ptr);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const Conv2DMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      const float* filter_ptr,
                      const float* bias_ptr,
                      ffStream_t stream);
  static void forward_kernel_wrapper(const Conv2DMeta* m,
                                     const float* input_ptr,
                                     float* output_ptr,
                                     const float* filter_ptr,
                                     const float* bias_ptr);
  static void backward_kernel(const Conv2DMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr,
                       ffStream_t stream);
  static void backward_kernel_wrapper(const Conv2DMeta* m,
                                      const float* input_ptr,
                                      float* input_grad_ptr,
                                      const float* output_ptr,
                                      float* output_grad_ptr,
                                      const float* kernel_ptr,
                                      float* kernel_grad_ptr,
                                      float* bias_grad_ptr);
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;
  bool estimate_sync_cost(Simulator* sim,
                          const MachineView& pc,
                          CostMetrics& cost_metrics) const override;

  void serialize(Legion::Serializer& s) const override;
  static PCG::Node deserialize(FFModel& ff, Legion::Deserializer& d, ParallelTensor inputs[], int num_inputs);

  static void construct_output_mappings(std::vector<ParallelDimMappingRecord> &);
  static void construct_mappings(std::vector<ParallelDimMappingRecord> &, bool use_bias);
  static void construct_weight_mappings(std::vector<ParallelDimMappingRecord> &, bool use_bias);

  std::unordered_map<std::pair<ParallelTensorShape, Conv2DParams>, Conv2D*> &get_cache(FFModel &ff) const;

  // size_t get_params_hash() const override;

  Conv2DParams get_params() const;

  tl::optional<RecordFormatter> as_dot() const override;
public:
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  ActiMode activation;
  int groups;
  bool use_bias;
};

}; // namespace FlexFlow

namespace std { 
  template <>
  struct hash<FlexFlow::Conv2DParams> {
    size_t operator()(FlexFlow::Conv2DParams const &) const;
  };
}; // namespace std


#endif // _FLEXFLOW_CONV_2D_H

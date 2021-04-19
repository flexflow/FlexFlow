#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "model.h"

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handler);
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
  bool relu, use_bias;
  char op_name[MAX_OPNAME];
};

class Conv2D : public Op {
public:
  Conv2D(FFModel& model,
         const Tensor input,
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
         const Tensor input,
         bool allocate_weights);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model);
  //Parameter* get_parameter(int index);
  //void create_weights(FFModel& model);
  //void create_input_partition(FFModel& model);

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
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
                      const float* bias_ptr);
  static void backward_kernel(const Conv2DMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;

/* #ifndef __CUDACC__ */
  void serialize(Legion::Serializer& s) const override;
  static Node deserialize(FFModel& ff, Legion::Deserializer& d, Tensor inputs[], int num_inputs);
/* #endif */ 
private:
  int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]); 
  int kernel_size(ParallelDim kernel_dims[MAX_TENSOR_DIM]); 
  int bias_size(ParallelDim bias_dims[MAX_TENSOR_DIM]); 

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();
public:
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
  bool use_bias;
  ActiMode activation;

  friend struct Conv2DModelCacheHash;
};

#endif // _FLEXFLOW_CONV_2D_H

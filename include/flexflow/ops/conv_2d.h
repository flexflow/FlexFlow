#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "flexflow/model.h"

namespace FlexFlow {
  
namespace Input {
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

namespace Output {
  enum {
    WIDTH = 0,
    HEIGHT = 1,
    CHANNEL = 2,
    SAMPLE = 3,
    REPLICA = 4,
    NUMDIM
  };
}

namespace Kernel {
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

namespace Bias {
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
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
  ActiMode activation;
  bool use_bias;

  bool is_valid(const Tensor input) const;
  void solve_dims(const Tensor input,
                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const;
  size_t get_hash(const Tensor input) const;
private:
  void mark_replica_dims(const Tensor input, 
                         ParallelDim output_dims[MAX_TENSOR_DIM],
                         ParallelDim kernel_dims[MAX_TENSOR_DIM],
                         ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
  int output_size(const Tensor input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const; 
  int kernel_size(const Tensor input,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM]) const; 
  int bias_size(const Tensor input,
                ParallelDim bias_dims[MAX_TENSOR_DIM]) const; 
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handler);
#ifdef LEGION_USE_CUDA
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
#endif
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
                      const float* bias_ptr,
                      cudaStream_t stream);
  static void backward_kernel(const Conv2DMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr,
                       cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  bool estimate_sync_cost(Simulator* sim,
                          const MachineView& pc,
                          CostMetrics& cost_metrics) const override;

  void serialize(Legion::Serializer& s) const override;
  static PCG::Node deserialize(FFModel& ff, Legion::Deserializer& d, Tensor inputs[], int num_inputs);

  static void construct_output_mappings(std::vector<ParallelDimMappingRecord> &);
  static void construct_mappings(std::vector<ParallelDimMappingRecord> &, bool use_bias);
  static void construct_weight_mappings(std::vector<ParallelDimMappingRecord> &, bool use_bias);

  size_t get_params_hash() const override;

  Conv2DParams get_params() const;
public:
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  ActiMode activation;
  int groups;
  bool use_bias;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_CONV_2D_H

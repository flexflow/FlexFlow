#ifndef _FLEXFLOW_POOL_2D_H
#define _FLEXFLOW_POOL_2D_H

#include "flexflow/model.h"

namespace FlexFlow {

namespace Pool2DInput {
  constexpr int NUMDIM = 5,
                WIDTH = 0,
                HEIGHT = 1,
                CHANNEL = 2,
                SAMPLE = 3,
                REPLICA = 4;
};

namespace Pool2DOutput {
  constexpr int NUMDIM = 5,
                WIDTH = 0,
                HEIGHT = 1,
                CHANNEL = 2,
                SAMPLE = 3,
                REPLICA = 4;
};

struct Pool2DParams {
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;

  bool is_valid(const ParallelTensor input) const;
  void solve_dims(const ParallelTensor input, 
                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims) const;
  size_t get_hash(const ParallelTensor input) const;
private:
  int output_size(const ParallelTensor input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const; 
};

class Pool2DMeta : public OpMeta {
public:
  Pool2DMeta(FFHandler handle);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenActivationDescriptor_t actiDesc;
  miopenPoolingDescriptor_t poolDesc;
#endif
  bool relu;
  char op_name[MAX_OPNAME];
};

class Pool2D : public Op {
public:
  Pool2D(FFModel& model,
         const ParallelTensor input,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, 
         ActiMode activation,
         const char* name);
  Pool2D(FFModel& model,
         Pool2D const &other,
         ParallelTensor const input);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void update(const FFModel&);
  void print_layer(const FFModel& model) override {assert(0);}
  static Op* create_operator_from_layer(
      FFModel& model,
      const Layer* layer,
      const std::vector<ParallelTensor>& inputs);

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(const Pool2DMeta* m,
                             const float* input_ptr,
                             float* output_ptr,
                             ffStream_t stream);
  static void backward_kernel(const Pool2DMeta* m,
                              const float* input_ptr,
                              float* input_grad_ptr,
                              const float* output_ptr,
                              const float* output_grad_ptr,
                              ffStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const override;

  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel& ff, Legion::Deserializer& d, ParallelTensor inputs[], int num_inputs);

  static void construct_output_mappings(std::vector<ParallelDimMappingRecord> &);

  Pool2DParams get_params() const;
  size_t get_params_hash() const override;
private:
  int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]); 

  void register_mappings();
  void register_output_mappings();
public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
};

}; // namespace FlexFlow

#endif //_FLEXFLOW_POOL_2D_H

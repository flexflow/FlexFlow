#ifndef _FLEXFLOW_POOL_2D_H
#define _FLEXFLOW_POOL_2D_H

#include "model.h"

struct Pool2DParams {
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
};

class Pool2DMeta : public OpMeta {
public:
  Pool2DMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  bool relu;
  char op_name[MAX_OPNAME];
};

class Pool2D : public Op {
public:
  Pool2D(FFModel& model,
         const Tensor input,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, 
         ActiMode activation,
         const char* name);
  Pool2D(FFModel& model,
         Pool2D const &other,
         Tensor const input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

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
                             float* output_ptr);
  static void backward_kernel(const Pool2DMeta* m,
                              const float* input_ptr,
                              float* input_grad_ptr,
                              const float* output_ptr,
                              const float* output_grad_ptr);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;

  void serialize(Legion::Serializer &) const override;
  static Node deserialize(FFModel& ff, Legion::Deserializer& d, Tensor inputs[], int num_inputs);
  static void construct_output_mappings(std::vector<ParallelDimMappingRecord> &);
  Pool2DParams get_params() const;
private:
  int output_size(ParallelDim output_dims[MAX_TENSOR_DIM]); 

  void register_mappings();
  void register_output_mappings();
public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
};

#endif //_FLEXFLOW_POOL_2D_H

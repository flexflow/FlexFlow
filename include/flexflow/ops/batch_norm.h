#ifndef _FLEXFLOW_BATCH_NORM_H
#define _FLEXFLOW_BATCH_NORM_H

#include "flexflow/model.h"

namespace FlexFlow {

class BatchNormMeta;
class BatchNorm : public Op {
public:
  BatchNorm(FFModel& model,
            const Tensor input,
            const Tensor scale,
            const Tensor bias,
            bool relu,
            const char* name);
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
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(BatchNormMeta *m,
                             float const *input_ptr,
                             float *output_ptr,
                             float const *scale_ptr,
                             float const *bias_ptr,
                             cudaStream_t stream);
  static void backward_kernel(BatchNormMeta *m,
                              float const *input_ptr,
                              float *output_grad_ptr,
                              float const *output_ptr,
                              float *input_grad_ptr,
                              float const *scale_ptr,
                              float *scale_grad_ptr,
                              float *bias_grad_ptr,
                              size_t numElements,
                              cudaStream_t stream);
public:
  bool relu;
  int num_replica;
};

class BatchNormMeta : public OpMeta {
public:
  BatchNormMeta(FFHandler handle,
                const BatchNorm* bn,
                Legion::Memory gpu_mem,
                int output_n,
                int output_c,
                int output_h,
                int output_w);
  ~BatchNormMeta(void);
  Realm::RegionInstance reserveInst;
  cudnnTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
};

}; // namespace FlexFlow

#endif

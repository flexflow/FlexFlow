#ifndef _FLEXFLOW_BATCH_NORM_H
#define _FLEXFLOW_BATCH_NORM_H

#include "flexflow/model.h"

namespace FlexFlow {

class BatchNormMeta;
class BatchNorm : public Op {
public:
  BatchNorm(FFModel& model,
            const ParallelTensor input,
            const ParallelTensor scale,
            const ParallelTensor bias,
            bool relu,
            const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void pipeinit(const FFModel&)  override {assert(0);}
  void pipeforward(const FFModel&)  override {assert(0);}
  void pipebackward(const FFModel&)  override {assert(0);}
  void update(const FFModel&);
  void print_layer(const FFModel& model) override {assert(0);}

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
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;
  static void forward_kernel(BatchNormMeta *m,
                             float const *input_ptr,
                             float *output_ptr,
                             float const *scale_ptr,
                             float const *bias_ptr);
                             //ffStream_t stream);
  static void backward_kernel(BatchNormMeta *m,
                              float const *input_ptr,
                              float *output_grad_ptr,
                              float const *output_ptr,
                              float *input_grad_ptr,
                              float const *scale_ptr,
                              float *scale_grad_ptr,
                              float *bias_grad_ptr,
                              size_t numElements);
                              //ffStream_t stream);
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
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnBatchNormMode_t mode;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  miopenActivationDescriptor_t actiDesc;
  miopenBatchNormMode_t mode;
#endif
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
};

}; // namespace FlexFlow

#endif

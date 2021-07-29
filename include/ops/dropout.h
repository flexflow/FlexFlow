#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "model.h"

class DropoutMeta;
class Dropout : public Op {
public:
  Dropout(FFModel& model,
          const Tensor input,
          float rate,
          unsigned long long seed,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
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
  static void forward_kernel(DropoutMeta *m,
                             float const *input_ptr,
                             float *output_ptr,
                             cudaStream_t stream);
  static void backward_kernel(DropoutMeta *m,
                              float const *output_grad_ptr,
                              float *input_grad_ptr,
                              cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  float rate;
  unsigned long long seed;
};

class DropoutMeta : public OpMeta {
public:
  DropoutMeta(FFHandler handle,
              const Dropout* dropout,
              Legion::Memory gpu_mem,
              const Legion::Domain& output_domain);
  ~DropoutMeta(void);
  Realm::RegionInstance reserveInst;
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnDropoutDescriptor_t dropoutDesc;
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
};

#endif

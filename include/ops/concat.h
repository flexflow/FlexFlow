#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "model.h"

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
  int axis;
};

class Concat : public Op {
public:
  Concat(FFModel& model,
         int n,
         const Tensor* inputs,
         int axis,
         const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  bool get_int_parameter(PMParameter, int*) const;
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  void init_meta(ConcatMeta *meta) const;
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(float* output,
                             float const * const *inputs,
                             int num_inputs,
                             int axis,
                             const Legion::Domain& out_domain,
                             const Legion::Domain* in_domain,
                             cudaStream_t stream);
  static void backward_kernel(const float* output_grad,
                              float** input_grads,
                              int num_inputs,
                              int axis,
                              const Legion::Domain& out_grad_domain,
                              const Legion::Domain* in_grad_domain,
                             cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;

  size_t get_params_hash() const override;
public:
  int axis;
};

#endif // _FLEXFLOW_CONCAT_H

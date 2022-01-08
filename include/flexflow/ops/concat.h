#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "flexflow/model.h"

namespace FlexFlow {

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
  int axis;
};

class Concat : public Op {
public:
  Concat(FFModel& model,
         int n,
         const ParallelTensor* inputs,
         int axis,
         const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  bool get_int_parameter(PMParameter, int*) const override;
  void print_layer(const FFModel& model) override {assert(0);}
  static Op* create_operator_from_layer(
      FFModel& model,
      const Layer* layer,
      const std::vector<ParallelTensor>& inputs);
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
                             ffStream_t stream);
  static void backward_kernel(const float* output_grad,
                              float** input_grads,
                              int num_inputs,
                              int axis,
                              const Legion::Domain& out_grad_domain,
                              const Legion::Domain* in_grad_domain,
                              ffStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const override;

  size_t get_params_hash() const override;
public:
  int axis;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_CONCAT_H

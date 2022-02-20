#ifndef _FLEXFLOW_AGGREGATE_SPEC_H_
#define _FLEXFLOW_AGGREGATE_SPEC_H_

#include "flexflow/model.h"

namespace FlexFlow {
  
#define AGGREGATE_SPEC_MAX_K 4
#define AGGREGATE_SPEC_MAX_BATCH_SIZE 32
#define AGGREGATE_SPEC_MAX_N 12

class AggregateSpecMeta : public OpMeta {
public:
  AggregateSpecMeta(FFHandler handle, int n);
  ~AggregateSpecMeta(void);
  float** dev_region_ptrs;
};

class AggregateSpec : public Op {
public:
  AggregateSpec(FFModel& model,
                const ParallelTensor* inputs,
                int _n, float _lambda_bal, const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void print_layer(const FFModel& model) override {assert(0);}
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel_wrapper(const AggregateSpecMeta *m, 
                                     float** exp_preds,
                                     const int* acc_gate_assign_ptr, 
                                     float* acc_output_ptr, 
                                     int n, const int k, int rows, 
                                     const int batch_size, int out_dim);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_kernel_wrapper(const AggregateSpecMeta *m, 
                                      float** exp_grads,
                                      const int* acc_gate_assign_ptr,
                                      const int* acc_true_gate_assign_ptr, 
                                      const float* acc_gate_pred_ptr, 
                                      float* acc_full_gate_grad_ptr,
                                      const float* acc_output_grad_ptr, 
                                      int n, const int k, int rows, 
                                      float lambda_bal,
                                      const int batch_size, int out_dim);
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;
public:
  int n;
  float lambda_bal;
};


}; // namespace FlexFlow
#endif

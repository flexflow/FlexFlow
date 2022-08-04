#ifndef _FLEXFLOW_AGGREGATE_H_
#define _FLEXFLOW_AGGREGATE_H_

#include "flexflow/model.h"

namespace FlexFlow {

#define AGGREGATE_MAX_K 4
#define AGGREGATE_MAX_BATCH_SIZE 32
#define AGGREGATE_MAX_N 12

class AggregateMeta : public OpMeta {
public:
  AggregateMeta(FFHandler handle, int n);
  ~AggregateMeta(void);
  float **dev_exp_preds;
  float **dev_exp_grads;
};

class Aggregate : public Op {
public:
  Aggregate(FFModel &model,
            ParallelTensor const *inputs,
            int _n,
            float _lambda_bal,
            char const *name);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_kernel_wrapper(AggregateMeta const *m,
                                     float **exp_preds,
                                     int const *acc_gate_assign_ptr,
                                     float const *acc_gate_pred_ptr,
                                     float *acc_output_ptr,
                                     int n,
                                     int const k,
                                     int rows,
                                     int const batch_size,
                                     int out_dim);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void backward_kernel_wrapper(AggregateMeta const *m,
                                      float **exp_preds,
                                      float **exp_grads,
                                      int const *acc_gate_assign_ptr,
                                      int const *acc_true_gate_assign_ptr,
                                      float const *acc_gate_pred_ptr,
                                      float *full_acc_gate_grad_ptr,
                                      float const *acc_output_grad_ptr,
                                      int n,
                                      int const k,
                                      int rows,
                                      float lambda_bal,
                                      int const batch_size,
                                      int out_dim);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;

public:
  int n;
  float lambda_bal;
};

}; // namespace FlexFlow

#endif

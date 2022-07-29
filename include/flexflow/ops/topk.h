#ifndef _FLEXFLOW_TOPK_H_
#define _FLEXFLOW_TOPK_H_

#include "flexflow/model.h"

namespace FlexFlow {

class TopKMeta : public OpMeta {
public:
  TopKMeta(FFHandler handle);
  bool sorted;
};

class TopK : public Op {
public:
  TopK(FFModel& model,
       const ParallelTensor input,
       int k, bool sorted,
       const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void reset_idx(const FFModel&) override {assert(0);}
  void pipeinit(const FFModel&)  override {assert(0);}
  void pipeforward(const FFModel&)  override {assert(0);}
  void pipebackward(const FFModel&)  override {assert(0);}
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
  static void forward_kernel(const TopKMeta* m,
                             const float* input_ptr,
                             float* output_ptr,
                             int* indices_ptr,
                             size_t batch_size, int length, int k,
                             bool sorted,
                             ffStream_t stream);
  static void forward_kernel_wrapper(const TopKMeta* m,
                                     const float* input_ptr,
                                     float* output_ptr,
                                     int* indices_ptr,
                                     size_t batch_size, int length, int k,
                                     bool sorted);
  static void backward_kernel(const TopKMeta* m,
                              const float* out_grad_ptr,
                              const int* indices_ptr,
                              float* in_grad_ptr,
                              size_t batch_size, int length, int k,
                              ffStream_t stream);
  static void backward_kernel_wrapper(const TopKMeta* m,
                                      const float* out_grad_ptr,
                                      const int* indices_ptr,
                                      float* in_grad_ptr,
                                      size_t batch_size, int length, int k);
public:
  int k;
  bool sorted;
};

}; // namespace FlexFlow

#endif

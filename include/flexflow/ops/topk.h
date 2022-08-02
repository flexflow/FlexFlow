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
  TopK(FFModel &model,
       const ParallelTensor input,
       int k,
       bool sorted,
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
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  static void forward_kernel(TopKMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             int *indices_ptr,
                             size_t batch_size,
                             int length,
                             int k,
                             bool sorted,
                             ffStream_t stream);
  static void forward_kernel_wrapper(TopKMeta const *m,
                                     float const *input_ptr,
                                     float *output_ptr,
                                     int *indices_ptr,
                                     size_t batch_size,
                                     int length,
                                     int k,
                                     bool sorted);
  static void backward_kernel(TopKMeta const *m,
                              float const *out_grad_ptr,
                              int const *indices_ptr,
                              float *in_grad_ptr,
                              size_t batch_size,
                              int length,
                              int k,
                              ffStream_t stream);
  static void backward_kernel_wrapper(TopKMeta const *m,
                                      float const *out_grad_ptr,
                                      int const *indices_ptr,
                                      float *in_grad_ptr,
                                      size_t batch_size,
                                      int length,
                                      int k);

public:
  int k;
  bool sorted;
};

}; // namespace FlexFlow

#endif

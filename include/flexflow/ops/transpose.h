#ifndef _FLEXFLOW_TRANSPOSE_H_
#define _FLEXFLOW_TRANSPOSE_H_

#include "flexflow/model.h"

namespace FlexFlow {

class TransposeMeta : public OpMeta {
public:
  TransposeMeta(FFHandler handler) : OpMeta(handler){};
  int num_dim;
  int perm[MAX_TENSOR_DIM];
};

class Transpose : public Op {
public:
  Transpose(FFModel &model,
            const ParallelTensor input,
            std::vector<int> const &perm,
            char const *name);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void reset_idx(FFModel const &) override {
    assert(0);
  }
  void pipeinit(FFModel const &) override {
    assert(0);
  }
  void pipeforward(FFModel const &) override {
    assert(0);
  }
  void pipebackward(FFModel const &) override {
    assert(0);
  }
  void print_layer(FFModel const &model) override {
    assert(0);
  }

  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  void init_meta(TransposeMeta *m,
                 Legion::Domain const &in_domain,
                 Legion::Domain const &out_domain) const;
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void forward_kernel(TransposeMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             Legion::Domain in_domain,
                             Legion::Domain out_domain,
                             ffStream_t stream);
  static void forward_kernel_wrapper(TransposeMeta const *m,
                                     float const *input_ptr,
                                     float *output_ptr,
                                     Legion::Domain in_domain,
                                     Legion::Domain out_domain);
  static void backward_kernel(TransposeMeta const *m,
                              float *input_grad_ptr,
                              float const *output_grad_ptr,
                              Legion::Domain in_grad_domain,
                              Legion::Domain out_grad_domain,
                              ffStream_t stream);
  static void backward_kernel_wrapper(TransposeMeta const *m,
                                      float *input_grad_ptr,
                                      float const *output_grad_ptr,
                                      Legion::Domain in_grad_domain,
                                      Legion::Domain out_grad_domain);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

public:
  int perm[MAX_TENSOR_DIM];
};

}; // namespace FlexFlow

#endif

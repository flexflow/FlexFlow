#ifndef _FLEXFLOW_TOPK_H_
#define _FLEXFLOW_TOPK_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/topk_params.h"

namespace FlexFlow {

class TopKMeta : public OpMeta {
public:
  TopKMeta(FFHandler handle);
  bool sorted;
};

class TopK : public Op {
public:
  using Params = TopKParams;
  using Input = ParallelTensor;
  TopK(FFModel &model,
       const ParallelTensor input,
       int k,
       bool sorted,
       char const *name);
  TopK(FFModel &model, TopK const &other, const ParallelTensor input);
  TopK(FFModel &model,
       Params const &params,
       Input const input,
       char const *name = nullptr);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
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
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
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
  Params get_params() const;

public:
  int k;
  bool sorted;
};

}; // namespace FlexFlow

#endif

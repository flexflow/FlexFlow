#ifndef _FLEXFLOW_ARG_TOPK_H_
#define _FLEXFLOW_ARG_TOPK_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/arg_topk_params.h"

namespace FlexFlow {

class ArgTopKMeta : public OpMeta {
public:
  ArgTopKMeta(FFHandler handle, Op const *op);
  bool sorted;
  int k;
  bool speculative_decoding;
};

class ArgTopK : public Op {
public:
  using Params = ArgTopKParams;
  using Input = ParallelTensor;
  ArgTopK(FFModel &model,
          LayerID const &layer_guid,
          const ParallelTensor input,
          int k,
          bool sorted,
          bool speculative_decoding,
          char const *name);
  ArgTopK(FFModel &model,
          LayerID const &layer_guid,
          ArgTopK const &other,
          const ParallelTensor input);
  ArgTopK(FFModel &model,
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
  static InferenceResult
      inference_task(Legion::Task const *task,
                     std::vector<Legion::PhysicalRegion> const &regions,
                     Legion::Context ctx,
                     Legion::Runtime *runtime);
  static BeamInferenceResult inference_speculative_task(
      Legion::Task const *task,
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
  template <typename DT>
  static void forward_kernel(ArgTopKMeta const *m,
                             DT const *input_ptr,
                             float *output_ptr,
                             int *indices_ptr,
                             size_t batch_size,
                             int length,
                             int k,
                             bool sorted,
                             BeamSearchBatchConfig const *bc,
                             ffStream_t stream);
  static void forward_kernel_wrapper(ArgTopKMeta const *m,
                                     GenericTensorAccessorR const &input,
                                     GenericTensorAccessorW const &prob,
                                     GenericTensorAccessorW const &indices,
                                     int batch_size,
                                     BeamSearchBatchConfig const *bc);
  Params get_params() const;

public:
  int k;
  bool sorted;
  bool speculative_decoding;
};

}; // namespace FlexFlow

#endif

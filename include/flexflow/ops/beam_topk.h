#ifndef _FLEXFLOW_BEAM_TOPK_H_
#define _FLEXFLOW_BEAM_TOPK_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/beam_topk_params.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class BeamTopKMeta : public OpMeta {
public:
  BeamTopKMeta(FFHandler handle,
               Op const *op,
               MemoryAllocator &gpu_mem_allocator);
  ~BeamTopKMeta(void);
  bool sorted;
  int max_beam_width;
  int *parent_ids;
  void *acc_probs;
  int *block_start_index;
  int *request_id;
  int *tokens_per_request;
  Realm::RegionInstance reserveInst;
};

class BeamTopK : public Op {
public:
  using Params = BeamTopKParams;
  using Input = ParallelTensor;
  BeamTopK(FFModel &model,
           const ParallelTensor input,
           LayerID const &_layer_guid,
           int max_beam_width,
           bool sorted,
           char const *name);
  BeamTopK(FFModel &model, BeamTopK const &other, const ParallelTensor input);
  BeamTopK(FFModel &model,
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
  static BeamInferenceResult
      inference_task(Legion::Task const *task,
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
  static void forward_kernel(BeamTopKMeta const *m,
                             BeamSearchBatchConfig const *bc,
                             DT const *input_ptr,
                             float *output_ptr,
                             int *indices_ptr,
                             int *parent_ptr,
                             int batch_size,
                             int length,
                             bool sorted,
                             ffStream_t stream);
  static void forward_kernel_wrapper(BeamTopKMeta const *m,
                                     BeamSearchBatchConfig const *bc,
                                     GenericTensorAccessorR const &input,
                                     float *output_ptr,
                                     int *indices_ptr,
                                     int *parent_ptr,
                                     int batch_size,
                                     int length,
                                     bool sorted);
  Params get_params() const;

public:
  bool sorted;
  int max_beam_width;
};

}; // namespace FlexFlow

#endif

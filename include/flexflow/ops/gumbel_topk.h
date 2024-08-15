#ifndef _FLEXFLOW_GUMBEL_TOPK_H_
#define _FLEXFLOW_GUMBEL_TOPK_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/gumbel_topk_params.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <curand.h>
#include <curand_kernel.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#endif
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class GumbelTopKMeta : public OpMeta {
public:
  bool sorted;
  int k;
  bool speculative_decoding;
  Realm::RegionInstance reserveInst;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  curandState *state;
  int state_max_length;
#elif defined(FF_USE_HIP_ROCM)
  hiprandState *state;
#endif
  GumbelTopKMeta(FFHandler handle,
                 Op const *op,
                 MemoryAllocator &gpu_mem_allocator);
  ~GumbelTopKMeta(void);
};

class GumbelTopK : public Op {
public:
  using Params = GumbelTopKParams;
  using Input = ParallelTensor;
  GumbelTopK(FFModel &model,
             LayerID const &layer_guid,
             ParallelTensor const input,
             int k,
             bool sorted,
             bool speculative_decoding,
             char const *name);
  GumbelTopK(FFModel &model,
             LayerID const &layer_guid,
             GumbelTopK const &other,
             ParallelTensor const input);
  GumbelTopK(FFModel &model,
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
  static InferenceResult inference_speculative_task(
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
  static void forward_kernel(GumbelTopKMeta const *m,
                             DT const *input_ptr,
                             float *log_probs_ptr,
                             float *perturbed_log_probs_ptr,
                             int *indices_ptr,
                             size_t batch_size,
                             int length,
                             int k,
                             bool sorted,
                             BatchConfig const *bc,
                             ffStream_t stream);
  static void
      forward_kernel_wrapper(GumbelTopKMeta const *m,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &log_probs,
                             GenericTensorAccessorW const &perturbed_log_probs,
                             GenericTensorAccessorW const &indices,
                             int batch_size,
                             BatchConfig const *bc);
  Params get_params() const;

public:
  int k;
  bool sorted;
  bool speculative_decoding;
};

}; // namespace FlexFlow

#endif

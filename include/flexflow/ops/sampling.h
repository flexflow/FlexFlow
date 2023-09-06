#ifndef _FLEXFLOW_SAMPLING_TOPK_H_
#define _FLEXFLOW_SAMPLING_TOPK_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/sampling_params.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <curand.h>
#include <curand_kernel.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#endif
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class SamplingMeta : public OpMeta {
public:
  float top_p;
  void *sorted_logits;
  int *sorted_idx;
  int *begin_offset;
  int *end_offset;
  int *idx;
  void *d_temp_storage;
  size_t temp_storage_bytes;
  Realm::RegionInstance reserveInst;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  curandState *state;
#elif defined(FF_USE_HIP_ROCM)
  hiprandState *state;
#endif
  SamplingMeta(FFHandler handle,
               Op const *op,
               int batch_size,
               int total_ele,
               GenericTensorAccessorW input,
               MemoryAllocator &gpu_mem_allocator);
  ~SamplingMeta(void);
};

class Sampling : public Op {
public:
  using Params = SamplingParams;
  using Input = ParallelTensor;
  Sampling(FFModel &model,
           const ParallelTensor input,
           float top_p,
           char const *name);
  Sampling(FFModel &model, Sampling const &other, const ParallelTensor input);
  Sampling(FFModel &model,
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
  static void forward_kernel(SamplingMeta const *m,
                             DT *input_ptr,
                             int *indices_ptr,
                             float top_p,
                             int length,
                             int batch_size,
                             ffStream_t stream);
  static void forward_kernel_wrapper(SamplingMeta const *m,
                                     GenericTensorAccessorW const &input,
                                     GenericTensorAccessorW const &indices,
                                     int batch_size);
  Params get_params() const;

public:
  float top_p;
};

}; // namespace FlexFlow

#endif
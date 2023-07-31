#ifndef _FLEXFLOW_ARG_MAX_H_
#define _FLEXFLOW_ARG_MAX_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/argmax_params.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class ArgMaxMeta : public OpMeta {
public:
  bool beam_search;
  float *probs;
  void *d_temp_storage;
  size_t temp_storage_bytes = 0;
  int *d_offsets;
  void *d_out;
  Realm::RegionInstance reserveInst;
  ArgMaxMeta(FFHandler handler,
             Op const *op,
             Legion::Domain const &input_domain,
             Legion::Domain const &output_domain,
             GenericTensorAccessorW input,
             int batch_size,
             int total_ele,
             MemoryAllocator &gpu_mem_allocator);
  ~ArgMaxMeta(void);
};

class ArgMax : public Op {
public:
  using Params = ArgMaxParams;
  using Input = ParallelTensor;
  ArgMax(FFModel &model,
         const ParallelTensor input,
         bool beam_search,
         char const *name);
  ArgMax(FFModel &model, ArgMax const &other, const ParallelTensor input);
  ArgMax(FFModel &model,
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
      inference_task_beam(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  static InferenceResult
      inference_task_norm(Legion::Task const *task,
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
  static void forward_kernel(ArgMaxMeta const *m,
                             DT *input_ptr,
                             int *indices_ptr,
                             float *prob_ptr,
                             int *parent_ptr,
                             int length,
                             int batch_size,
                             ffStream_t stream);
  static void forward_kernel_wrapper(ArgMaxMeta const *m,
                                     GenericTensorAccessorW const &input,
                                     GenericTensorAccessorW const &indices,
                                     GenericTensorAccessorW const &parent,
                                     int batch_size);
  Params get_params() const;

public:
  bool beam_search;
};

}; // namespace FlexFlow

#endif
#pragma once

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/utils/memory_allocator.h"
namespace FlexFlow {

class SigmoidSiluMultiMeta;

class SigmoidSiluMulti : public Op {
public:
  using Params = SigmoidSiluMultiParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;
  SigmoidSiluMulti(FFModel &model,
                   Params const &params,
                   Input const &inputs,
                   char const *name = nullptr);
  SigmoidSiluMulti(FFModel &model,
                   LayerID const &_layer_guid,
                   const ParallelTensor _input1,
                   const ParallelTensor _input2,
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
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  SigmoidSiluMultiParams get_params() const;

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  template <typename T>
  static void inference_kernel(SigmoidSiluMultiMeta const *m,
                               int num_elements,
                               T const *input1_ptr,
                               T const *input2_ptr,
                               T *output_ptr,
                               ffStream_t stream);
  static void inference_kernel_wrapper(SigmoidSiluMultiMeta const *m,
                                       GenericTensorAccessorR const &input1,
                                       GenericTensorAccessorR const &input2,
                                       GenericTensorAccessorW const &output);
};

class SigmoidSiluMultiMeta : public OpMeta {
public:
  SigmoidSiluMultiMeta(FFHandler handle,
                       SigmoidSiluMulti const *ln,
                       MemoryAllocator &gpu_mem_allocator);
  ~SigmoidSiluMultiMeta(void);

public:
  Realm::RegionInstance reserveInst;
};

}; // namespace FlexFlow

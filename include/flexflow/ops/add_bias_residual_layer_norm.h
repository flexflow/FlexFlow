#pragma once

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/utils/memory_allocator.h"
namespace FlexFlow {

class AddBiasResidualLayerNormMeta;

class AddBiasResidualLayerNorm : public Op {
public:
  using Params = AddBiasResidualLayerNormParams;
  using Input = ParallelTensor;
  AddBiasResidualLayerNorm(FFModel &model,
                           AddBiasResidualLayerNormParams const &params,
                           ParallelTensor input,
                           char const *name = nullptr,
                           bool allocate_weights = false);
  AddBiasResidualLayerNorm(FFModel &model,
                           LayerID const &_layer_guid,
                           const ParallelTensor _input,
                           std::vector<int> const &axes,
                           bool _elementwise_affine,
                           bool _use_bias,
                           float _eps,
                           bool allocate_weights,
                           char const *name);
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
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  // size_t get_params_hash() const override;
  AddBiasResidualLayerNormParams get_params() const;

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
  static void inference_kernel(AddBiasResidualLayerNormMeta const *m,
                               T const *input_ptr,
                               T *output_ptr,
                               T const *gamma_ptr,
                               T const *beta_ptr,
                               ffStream_t stream);
  static void inference_kernel_wrapper(AddBiasResidualLayerNormMeta const *m,
                                       GenericTensorAccessorR const &input,
                                       GenericTensorAccessorW &output,
                                       GenericTensorAccessorR const &gamma,
                                       GenericTensorAccessorR const &beta);

public:
  bool elementwise_affine, use_bias;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  std::vector<int> axes;
};

class AddBiasResidualLayerNormMeta : public OpMeta {
public:
  AddBiasResidualLayerNormMeta(FFHandler handle,
                               AddBiasResidualLayerNorm const *ln,
                               MemoryAllocator &gpu_mem_allocator);
  ~AddBiasResidualLayerNormMeta(void);

public:
  bool elementwise_affine, use_bias;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  void *mean_ptr, *rstd_ptr, *ds_ptr, *db_ptr, *scale_ptr, *bias_ptr;
  char op_name[MAX_OPNAME];
  Realm::RegionInstance reserveInst;
};

}; // namespace FlexFlow

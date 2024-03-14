#pragma once

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/utils/memory_allocator.h"
namespace FlexFlow {

class ResidualLayerNormMeta;

class ResidualLayerNorm : public Op {
public:
  using Params = ResidualLayerNormParams;
  using Input = std::tuple<ParallelTensor, ParallelTensor, ParallelTensor>;
  ResidualLayerNorm(FFModel &model,
                    Params const &params,
                    Input const &inputs,
                    bool allocate_weights = false,
                    char const *name = nullptr);
  ResidualLayerNorm(FFModel &model,
                    LayerID const &_layer_guid,
                    const ParallelTensor _input,
                    const ParallelTensor _residual1,
                    const ParallelTensor _residual2,
                    bool _use_two_residuals,
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
  ResidualLayerNormParams get_params() const;

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
  static void inference_kernel(ResidualLayerNormMeta const *m,
                               T const *input_ptr,
                               T const *residual1_ptr,
                               T const *residual2_ptr,
                               T *added_output_ptr,
                               T *output_ptr,
                               T const *gamma_ptr,
                               T const *beta_ptr,
                               ffStream_t stream);
  static void inference_kernel_wrapper(ResidualLayerNormMeta const *m,
                                       GenericTensorAccessorR const &input,
                                       GenericTensorAccessorR const &residual1,
                                       GenericTensorAccessorR const &residual2,
                                       GenericTensorAccessorW &added_output,
                                       GenericTensorAccessorW &output,
                                       GenericTensorAccessorR const &gamma,
                                       GenericTensorAccessorR const &beta);

public:
  bool elementwise_affine, use_bias, use_two_residuals;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  std::vector<int> axes;
};

class ResidualLayerNormMeta : public OpMeta {
public:
  ResidualLayerNormMeta(FFHandler handle,
                        ResidualLayerNorm const *ln,
                        MemoryAllocator &gpu_mem_allocator);
  ~ResidualLayerNormMeta(void);

public:
  bool elementwise_affine, use_bias, use_two_residuals;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  void *mean_ptr, *rstd_ptr, *ds_ptr, *db_ptr, *scale_ptr, *bias_ptr;
  Realm::RegionInstance reserveInst;
};

}; // namespace FlexFlow

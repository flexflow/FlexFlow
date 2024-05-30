#pragma once

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/utils/memory_allocator.h"
namespace FlexFlow {

class LayerNormMeta;

class LayerNorm : public Op {
public:
  using Params = LayerNormParams;
  using Input = ParallelTensor;
  LayerNorm(FFModel &model,
            LayerNormParams const &params,
            ParallelTensor input,
            char const *name = nullptr,
            bool allocate_weights = false);
  LayerNorm(FFModel &model,
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
  LayerNormParams get_params() const;

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  template <typename T>
  static void forward_kernel(LayerNormMeta const *m,
                             T const *input_ptr,
                             T *output_ptr,
                             T const *gamma_ptr,
                             T const *beta_ptr,
                             ffStream_t stream);
  static void forward_kernel_wrapper(LayerNormMeta const *m,
                                     GenericTensorAccessorR const &input,
                                     GenericTensorAccessorW &output,
                                     GenericTensorAccessorR const &gamma,
                                     GenericTensorAccessorR const &beta);
  template <typename T>
  static void backward_kernel(LayerNormMeta const *m,
                              T const *output_grad_ptr,
                              T const *input_ptr,
                              T *input_grad_ptr,
                              T const *gamma_ptr,
                              T *gamma_grad_ptr,
                              T *beta_grad_ptr,
                              ffStream_t stream);
  template <typename T>
  static void backward_kernel_wrapper(LayerNormMeta const *m,
                                      T const *output_grad_ptr,
                                      T const *input_ptr,
                                      T *input_grad_ptr,
                                      T const *gamma_ptr,
                                      T *gamma_grad_ptr,
                                      T *beta_grad_ptr);

public:
  bool elementwise_affine, use_bias;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  std::vector<int> axes;
};

class LayerNormMeta : public OpMeta {
public:
  LayerNormMeta(FFHandler handle,
                LayerNorm const *ln,
                MemoryAllocator &gpu_mem_allocator);
  ~LayerNormMeta(void);

public:
  bool elementwise_affine, use_bias;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  void *mean_ptr, *rstd_ptr, *ds_ptr, *db_ptr, *scale_ptr, *bias_ptr;
  Realm::RegionInstance reserveInst;
};

}; // namespace FlexFlow

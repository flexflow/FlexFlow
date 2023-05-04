#pragma once

#include "flexflow/model.h"
#include "flexflow/ops/reduce_params.h"

namespace FlexFlow {

class Reduce;

class ReduceMeta : public OpMeta {
public:
  ReduceMeta(FFHandler handler,
             Reduce const *rd,
             Legion::Domain const &input_domain);
  ~ReduceMeta(void);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnReduceTensorDescriptor_t reduceDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenReduceTensorDescriptor_t reduceDesc;
#endif
};

class Reduce : public Op {
public:
  using Params = ReduceParams;
  using Input = ParallelTensor;
  Reduce(FFModel &model,
         Params const &params,
         const Input input,
         char const *name = nullptr);
  Reduce(FFModel &model,
         LayerID const &layer_guid,
         const ParallelTensor input,
         std::vector<int> const &axes,
         bool keepdims,
         char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
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
  static void forward_kernel(ReduceMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             ffStream_t stream);
  static void forward_kernel_wrapper(ReduceMeta const *m,
                                     GenericTensorAccessorR const &input,
                                     GenericTensorAccessorW const &output);
  static void backward_kernel(ReduceMeta const *m,
                              float const *output_grad_ptr,
                              float *input_grad_ptr,
                              ffStream_t stream);
  static void backward_kernel_wrapper(ReduceMeta const *m,
                                      GenericTensorAccessorR const &output_grad,
                                      GenericTensorAccessorW const &input_grad);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  Params get_params() const;

public:
  int num_axes;
  int axes[MAX_TENSOR_DIM];
  bool keepdims;
};

}; // namespace FlexFlow

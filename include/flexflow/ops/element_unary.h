#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/element_unary_params.h"

namespace FlexFlow {

class ElementUnaryMeta : public OpMeta {
public:
  ElementUnaryMeta(FFHandler handle);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
  OperatorType op_type;
  DataType data_type;
  bool inplace;
  float scalar;
};

class ElementUnary : public Op {
public:
  using Params = ElementUnaryParams;
  using Input = ParallelTensor;

  ElementUnary(FFModel &model,
               LayerID const &layer_guid,
               OperatorType type,
               const ParallelTensor x,
               bool inplace,
               char const *name,
               float scalar);
  ElementUnary(FFModel &model,
               Params const &params,
               Input const x,
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
  void map_output_tensors(FFModel &model) override;
  bool can_inplace_output() override;
  bool has_inplace_output() override;
  void do_inplace_output() override;
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
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename T>
  static void
      forward_task_with_type(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  template <typename T>
  static void backward_task_with_type(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  static void init_kernel(ElementUnaryMeta *m,
                          Legion::Domain const &input_domain,
                          Legion::Domain const &output_domain);
  template <typename T>
  static void forward_kernel(ElementUnaryMeta const *m,
                             T const *in_ptr,
                             T *out_ptr,
                             size_t num_elements,
                             ffStream_t stream);
  template <typename T>
  static void forward_kernel_wrapper(ElementUnaryMeta const *m,
                                     T const *in_ptr,
                                     T *out_ptr,
                                     size_t num_elements);
  template <typename T>
  static void backward_kernel(ElementUnaryMeta const *m,
                              T const *in_ptr,
                              T *in_grad_ptr,
                              T const *out_ptr,
                              T const *out_grad_ptr,
                              size_t num_elements,
                              ffStream_t stream);
  template <typename T>
  static void backward_kernel_wrapper(ElementUnaryMeta const *m,
                                      T const *in_ptr,
                                      T *in_grad_ptr,
                                      T const *out_ptr,
                                      T const *out_grad_ptr,
                                      size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  static bool use_cudnn(OperatorType type);

  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;

  Params get_params() const;

private:
  bool inplace;

public:
  float scalar;
};

}; // namespace FlexFlow

#endif // _ELEMENT_UNARY_H

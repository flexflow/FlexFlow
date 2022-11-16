#ifndef _STOP_GRAD_H
#define _STOP_GRAD_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/stop_grad_params.h"

namespace FlexFlow {

class StopGradMeta : public OpMeta {
public:
  StopGradMeta(FFHandler handle);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
  DataType data_type;
  char op_name[MAX_OPNAME];
};

class StopGrad : public Op {
public:
  using Params = StopGradParams;
  using Input = ParallelTensor;

  StopGrad(FFModel &model,
               const ParallelTensor x,
               char const *name);
  StopGrad(FFModel &model,
               Params const &params,
               Input const x,
               char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  void map_output_tensors(FFModel &model) override;
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
  static void init_kernel(StopGradMeta *m,
                          Legion::Domain const &input_domain,
                          Legion::Domain const &output_domain);
  template <typename T>
  static void forward_kernel(StopGradMeta const *m,
                             const T *in_ptr,
                             T *out_ptr,
                             size_t num_elements,
                             ffStream_t stream);
  template <typename T>
  static void forward_kernel_wrapper(StopGradMeta const *m,
                                     const T *in_ptr,
                                     T *out_ptr,
                                     size_t num_elements);
  template <typename T>
  static void backward_kernel(StopGradMeta const *m,
                              const T *in_ptr,
                              T *in_grad_ptr,
                              const T *out_ptr,
                              const T *out_grad_ptr,
                              size_t num_elements,
                              ffStream_t stream);
  template <typename T>
  static void backward_kernel_wrapper(StopGradMeta const *m,
                                      const T *in_ptr,
                                      T *in_grad_ptr,
                                      const T *out_ptr,
                                      const T *out_grad_ptr,
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

#endif // _STOP_GRAD_H

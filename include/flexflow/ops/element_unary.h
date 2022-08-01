#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "flexflow/model.h"

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
  ElementUnary(FFModel &model,
               OperatorType type,
               const ParallelTensor x,
               bool inplace,
               const char *name,
               float scalar);
  void init(const FFModel &) override;
  void forward(const FFModel &) override;
  void backward(const FFModel &) override;
  void print_layer(const FFModel &model) override { assert(0); }
  bool can_inplace_output() override;
  bool has_inplace_output() override;
  void do_inplace_output() override;
  static Op *
  create_operator_from_layer(FFModel &model,
                             const Layer *layer,
                             const std::vector<ParallelTensor> &inputs);

  static OpMeta *init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename T>
  static void
  forward_task_with_type(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  template <typename T>
  static void
  backward_task_with_type(const Legion::Task *task,
                          const std::vector<Legion::PhysicalRegion> &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  static void init_kernel(ElementUnaryMeta *m,
                          const Legion::Domain &input_domain,
                          const Legion::Domain &output_domain);
  template <typename T>
  static void forward_kernel(const ElementUnaryMeta *m,
                             const T *in_ptr,
                             T *out_ptr,
                             size_t num_elements,
                             ffStream_t stream);
  template <typename T>
  static void forward_kernel_wrapper(const ElementUnaryMeta *m,
                                     const T *in_ptr,
                                     T *out_ptr,
                                     size_t num_elements);
  template <typename T>
  static void backward_kernel(const ElementUnaryMeta *m,
                              const T *in_ptr,
                              T *in_grad_ptr,
                              const T *out_ptr,
                              const T *out_grad_ptr,
                              size_t num_elements,
                              ffStream_t stream);
  template <typename T>
  static void backward_kernel_wrapper(const ElementUnaryMeta *m,
                                      const T *in_ptr,
                                      T *in_grad_ptr,
                                      const T *out_ptr,
                                      const T *out_grad_ptr,
                                      size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             const MachineView &pc,
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

  size_t get_params_hash() const override;

private:
  bool inplace;

public:
  float scalar;
};

}; // namespace FlexFlow

#endif // _ELEMENT_UNARY_H

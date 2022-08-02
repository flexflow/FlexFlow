#ifndef _FLEXFLOW_ELEMENT_BINARY_H
#define _FLEXFLOW_ELEMENT_BINARY_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"

namespace FlexFlow {

struct ElementBinaryParams {
  OperatorType type;

  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(ElementBinaryParams const &, ElementBinaryParams const &);

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t input1Tensor, input2Tensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
  cudnnReduceTensorDescriptor_t reduceAddDesc;
#else
  miopenTensorDescriptor_t input1Tensor, input2Tensor, outputTensor;
  miopenTensorOp_t opDesc;
  miopenReduceTensorDescriptor_t reduceAddDesc;
#endif
  OperatorType op_type;
  bool inplace_a, has_same_operands;
  bool broadcast_input1, broadcast_input2;
};

class ElementBinary : public Op {
public:
  using Params = ElementBinaryParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;

  ElementBinary(FFModel &model,
                OperatorType type,
                const ParallelTensor x,
                const ParallelTensor y,
                bool inplace_a,
                char const *name);
  ElementBinary(FFModel &model,
                Params const &params,
                Input const &inputs,
                char const *name = nullptr,
                bool inplace_a = false);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
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
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void init_kernel(ElementBinaryMeta *m,
                          Legion::Domain const &input1_domain,
                          Legion::Domain const &input2_domain,
                          Legion::Domain const &output_domain);
  static void forward_kernel(ElementBinaryMeta const *m,
                             float const *in1_ptr,
                             float const *in2_ptr,
                             float *out_ptr,
                             ffStream_t stream);
  static void forward_kernel_wrapper(ElementBinaryMeta const *m,
                                     float const *in1_ptr,
                                     float const *in2_ptr,
                                     float *out_ptr);
  static void backward_kernel(ElementBinaryMeta const *m,
                              float const *out_grad_ptr,
                              float const *in1_ptr,
                              float const *in2_ptr,
                              float *in1_grad_ptr,
                              float *in2_grad_ptr,
                              ffStream_t stream);
  static void backward_kernel_wrapper(ElementBinaryMeta const *m,
                                      float const *out_grad_ptr,
                                      float const *in1_ptr,
                                      float const *in2_ptr,
                                      float *in1_grad_ptr,
                                      float *in2_grad_ptr);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  Params get_params() const;

public:
  bool inplace_a, has_same_operands;
  bool broadcast_input1, broadcast_input2;
};

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ElementBinaryParams> {
  size_t operator()(FlexFlow::ElementBinaryParams const &) const;
};
}; // namespace std

#endif // _FLEXFFLOW_ELEMENT_BINARY_H

#ifndef _FLEXFLOW_ELEMENT_BINARY_H
#define _FLEXFLOW_ELEMENT_BINARY_H

#include "flexflow/model.h"

namespace FlexFlow {

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
#endif
  OperatorType op_type;
  bool inplace_a;
};

class ElementBinary : public Op {
public:
  ElementBinary(FFModel& model,
                OperatorType type,
                const Tensor x,
                const Tensor y,
                bool inplace_a,
                const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  bool can_inplace_output();
  bool has_inplace_output();
  void do_inplace_output();

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(const ElementBinaryMeta* m,
                      const float* in1_ptr,
                      const float* in2_ptr,
                      float* out_ptr,
                      cudaStream_t stream);
  static void backward_kernel(const ElementBinaryMeta* m,
                       const float* out_grad_ptr,
                       const float* in1_ptr,
                       const float* in2_ptr,
                       float* in1_grad_ptr,
                       float* in2_grad_ptr,
                       cudaStream_t stream);
  size_t get_params_hash() const override;
public:
  bool inplace_a;
};

}; // namespace FlexFlow

#endif // _FLEXFFLOW_ELEMENT_BINARY_H

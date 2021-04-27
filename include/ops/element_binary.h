#ifndef _FLEXFLOW_ELEMENT_BINARY_H
#define _FLEXFLOW_ELEMENT_BINARY_H

#include "model.h"

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
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
                      float* out_ptr);
  static void backward_kernel(const ElementBinaryMeta* m,
                       const float* out_grad_ptr,
                       const float* in1_ptr,
                       const float* in2_ptr,
                       float* in1_grad_ptr,
                       float* in2_grad_ptr);

  size_t get_params_hash() const override;
public:
  bool inplace_a;
};

#endif // _FLEXFFLOW_ELEMENT_BINARY_H

#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "model.h"

class ElementUnaryMeta : public OpMeta {
public:
  ElementUnaryMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  OperatorType op_type;
  bool inplace;
  float scalar;
};

class ElementUnary : public Op {
public:
  ElementUnary(FFModel& model,
               OperatorType type,
               const Tensor x,
               bool inplace,
               const char* name,
	       float scalar);
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
  static void forward_kernel(const ElementUnaryMeta* m,
                      const float* in_ptr,
                      float* out_ptr,
                      size_t num_elements,
                      cudaStream_t stream);
  static void backward_kernel(const ElementUnaryMeta* m,
                       const float* in_ptr,
                       float* in_grad_ptr,
                       const float* out_ptr,
                       const float* out_grad_ptr,
                       size_t num_elements,
                       cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static bool use_cudnn(OperatorType type);

  void serialize(Legion::Serializer&) const override;
  static Node deserialize(FFModel& ff, Legion::Deserializer& d, Tensor inputs[], int num_inputs);
  Op *materialize(FFModel& ff, Tensor inputs[], int num_inputs) const override;

  size_t get_params_hash() const override;
public:
  float scalar;
private:
  bool inplace;
};

#endif // _ELEMENT_UNARY_H

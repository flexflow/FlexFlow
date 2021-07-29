#ifndef _FLEXFLOW_SOFTMAX_H
#define _FLEXFLOW_SOFTMAX_H

#include "model.h"

class Softmax;

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle,
              const Softmax* softmax,
              const Legion::Domain& input_domain);
  cudnnTensorDescriptor_t inputTensor;
  bool profiling;
  int dim;
  char op_name[MAX_OPNAME];
};

class Softmax : public Op {
public:
  Softmax(FFModel& model,
          const Tensor logit,
          int dim,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  bool get_int_parameter(PMParameter, int*) const;
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  void init_meta(SoftmaxMeta *m,
                 Legion::Rect<2> const &input,
                 Legion::Rect<2> const &output) const;
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(SoftmaxMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             cudaStream_t stream);
  static void backward_kernel(float *input_grad_ptr,
                              float const *output_grad_ptr,
                              size_t num_elements,
                              cudaStream_t stream);
  size_t get_params_hash() const override;
private:
  template<int NDIM>
  static void forward_task_with_dim(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx, Legion::Runtime *runtime);
  template<int NDIM>
  static void backward_task_with_dim(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &regions,
                                     Legion::Context ctx, Legion::Runtime *runtime);
public:
  int dim;
};

#endif // _FLEXFLOW_SOFTMAX_H

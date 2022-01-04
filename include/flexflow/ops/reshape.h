#ifndef _FLEXFLOW_RESHAPE_H
#define _FLEXFLOW_RESHAPE_H

#include "flexflow/model.h"

namespace FlexFlow {

class Reshape : public Op {
public:
  Reshape(FFModel& model,
          const ParallelTensor input,
          const std::vector<int>& shape,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
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
  template<typename T>
  static void forward_kernel(const T* input_ptr,
                             T* output_ptr,
                             size_t num_elements,
                             cudaStream_t stream);
  template<typename T>
  static void backward_kernel(T* input_grad_ptr,
                              const T* output_grad_ptr,
                              size_t num_elements,
                              cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
};

}; // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "flexflow/model.h"

namespace FlexFlow {

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle) {};
};

class Flat : public Op {
public:
  Flat(FFModel& model,
       const Tensor input,
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
  static void forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements,
                             cudaStream_t stream);
  static void backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements,
                              cudaStream_t stream);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  Legion::Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx) const;

  void serialize(Legion::Serializer&) const override;
  static PCG::Node deserialize(FFModel& ff, Legion::Deserializer& d, Tensor inputs[], int num_inputs);
  Op *materialize(FFModel& ff, Tensor inputs[], int num_inputs) const override;
  static void construct_output_mappings(std::vector<ParallelDimMappingRecord> &);

  size_t get_params_hash() const override;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_FLAT_H

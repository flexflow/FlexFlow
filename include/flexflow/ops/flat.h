#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "flexflow/model.h"

namespace FlexFlow {
  
namespace Input {
  constexpr int NUMDIM = 5,
                WIDTH = 0,
                HEIGHT = 1,
                CHANNEL = 2,
                SAMPLE = 3,
                REPLICA = 4;
}

namespace Output {
  constexpr int NUMDIM = 3,
                CHANNEL = 0,
                SAMPLE = 1,
                REPLICA = 2;
}

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle) {};
};

class Flat : public Op {
public:
  Flat(FFModel& model,
       const ParallelTensor input,
       const char* name);

  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void print_layer(const FFModel& model) override {assert(0);}
  static Op* create_operator_from_layer(FFModel& model,
                                        const Layer* layer,
                                        const std::vector<ParallelTensor>& inputs);

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  static void forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements,
                             cudaStream_t stream);
  static void backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements,
                              cudaStream_t stream);
#else
  static void forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements,
                             hipStream_t stream);
  static void backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements,
                              hipStream_t stream);
#endif
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const override;
  Legion::Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx) const override;

  void serialize(Legion::Serializer&) const override;
  static PCG::Node deserialize(FFModel& ff, Legion::Deserializer& d, ParallelTensor inputs[], int num_inputs);
  Op *materialize(FFModel& ff, ParallelTensor inputs[], int num_inputs) const override;
  static void construct_output_mappings(std::vector<ParallelDimMappingRecord> &);

  size_t get_params_hash() const override;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_FLAT_H

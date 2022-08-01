#ifndef _FLEXFLOW_RESHAPE_H
#define _FLEXFLOW_RESHAPE_H

#include "flexflow/model.h"

namespace FlexFlow {

class ReshapeMeta : public OpMeta {
public:
  ReshapeMeta(FFHandler handler);
  DataType data_type;
};

struct ReshapeParams {
  ReshapeParams(const std::vector<int> &_shape);
  size_t get_hash(const ParallelTensor input) const;
  std::vector<int> shape;
};

class Reshape : public Op {
public:
  Reshape(FFModel &model,
          const ParallelTensor input,
          const std::vector<int> &shape,
          const char *name);
  void init(const FFModel &) override;
  void forward(const FFModel &) override;
  void backward(const FFModel &) override;
  void print_layer(const FFModel &model) override { assert(0); }
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
  static void forward_kernel(const T *input_ptr,
                             T *output_ptr,
                             size_t num_elements,
                             ffStream_t stream);
  template <typename T>
  static void forward_kernel_wrapper(const T *input_ptr,
                                     T *output_ptr,
                                     size_t num_elements);
  template <typename T>
  static void backward_kernel(T *input_grad_ptr,
                              const T *output_grad_ptr,
                              size_t num_elements,
                              ffStream_t stream);
  template <typename T>
  static void backward_kernel_wrapper(T *input_grad_ptr,
                                      const T *output_grad_ptr,
                                      size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             const MachineView &pc,
                             CostMetrics &cost_metrics) const override;
  size_t get_params_hash() const override;
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  ReshapeParams get_params() const;

public:
  size_t shape_length;
  int shape_array[MAX_TENSOR_DIM];
};

}; // namespace FlexFlow

#endif

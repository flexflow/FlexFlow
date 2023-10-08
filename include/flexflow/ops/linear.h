#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "flexflow/inference.h"
#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/ops/linear_params.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class FFModel;
class Layer;

class Linear : public Op {
public:
  using Params = LinearParams;
  using Input = ParallelTensor;

  Linear(FFModel &model,
         LayerID const &layer_guid,
         const ParallelTensor input,
         int out_dim,
         ActiMode activation,
         RegularizerMode kernel_reg_type,
         float kernel_reg_lambda,
         bool _use_bias,
         DataType _data_type,
         DataType _quantization_type,
         bool offload,
         bool allocate_weights,
         char const *name);
  Linear(FFModel &model,
         Linear const &other,
         ParallelTensor const input,
         bool allocate_weights);
  Linear(FFModel &model,
         LinearParams const &params,
         ParallelTensor input,
         char const *name = nullptr,
         bool allocate_weights = false);

  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  void print_layer(FFModel const &model) override;
  bool get_int_parameter(PMParameter, int *) const override;
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
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
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  bool estimate_sync_cost(Simulator *sim,
                          MachineView const &pc,
                          CostMetrics &cost_metrics) const override;
  ParallelConfig get_random_parallel_config(FFModel const &ff) const override;
  bool is_valid_parallel_config(FFModel const &ff,
                                ParallelConfig const &pc) const override;

  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  // size_t get_params_hash() const override;
  LinearParams get_params() const;

private:
  Linear(int guid,
         bool profiling,
         bool inference_debugging,
         const ParallelTensor input,
         int out_dim,
         ActiMode activation,
         bool use_bias,
         bool allocate_weights,
         char const *name);

  template <typename DT, typename WT, int NDIM>
  static OpMeta *
      init_task_with_dim(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  template <typename DT, typename WT, int NDIM>
  static void
      forward_task_with_dim(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename DT, int NDIM>
  static void
      backward_task_with_dim(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();

public:
  int in_channels, out_channels;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  bool use_bias;
  ParallelTensor replica;
  DataType quantization_type;
  bool offload;
};

}; // namespace FlexFlow

#endif // _FLEXLOW_LINEAR_H

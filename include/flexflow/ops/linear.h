#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/linear_params.h"

namespace FlexFlow {

class FFModel;
class Layer;

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle, int batch_size);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#else
  miopenTensorDescriptor_t outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
  float const *one_ptr;
  ActiMode activation;
  bool use_bias;
  DataType input_type, weight_type, output_type;
  char op_name[MAX_OPNAME];
};

class Linear : public Op {
public:
  using Params = LinearParams;
  using Input = ParallelTensor;

  Linear(FFModel &model,
         LayerID const &layer_guid,
         const ParallelTensor input,
         int out_dim,
         ActiMode activation,
         bool _use_bias,
         DataType _data_type,
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
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void reset_idx(FFModel const &) override;
  void pipeinit(FFModel const &) override;
  void pipeforward(FFModel const &) override;
  void pipebackward(FFModel const &) override;
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
  static void init_kernel(LinearMeta *m,
                          void const *weight_ptr,
                          int batch_size,
                          int channel);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void forward_kernel(LinearMeta const *m,
                             void const *input_ptr,
                             void *output_ptr,
                             void const *filter_ptr,
                             void const *bias_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size,
                             ffStream_t stream);
  static void forward_kernel_wrapper(LinearMeta const *m,
                                     void const *input_ptr,
                                     void *output_ptr,
                                     void const *filter_ptr,
                                     void const *bias_ptr,
                                     int in_dim,
                                     int out_dim,
                                     int batch_size);
  static void backward_kernel(LinearMeta const *m,
                              void const *input_ptr,
                              void *input_grad_ptr,
                              void const *output_ptr,
                              void *output_grad_ptr,
                              void const *kernel_ptr,
                              void *kernel_grad_ptr,
                              void *bias_ptr,
                              int in_dim,
                              int out_dim,
                              int batch_size,
                              ffStream_t stream);
  static void backward_kernel_wrapper(LinearMeta const *m,
                                      void const *input_ptr,
                                      void *input_grad_ptr,
                                      void const *output_ptr,
                                      void *output_grad_ptr,
                                      void const *kernel_ptr,
                                      void *kernel_grad_ptr,
                                      void *bias_ptr,
                                      int in_dim,
                                      int out_dim,
                                      int batch_size);
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
         const ParallelTensor input,
         int out_dim,
         ActiMode activation,
         bool use_bias,
         bool allocate_weights,
         char const *name);

  template <int NDIM>
  static OpMeta *
      init_task_with_dim(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  template <int NDIM>
  static void
      forward_task_with_dim(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <int NDIM>
  static void
      backward_task_with_dim(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static bool use_activation(ActiMode mode);

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();

public:
  int in_channels, out_channels;
  ActiMode activation;
  bool use_bias;
  ParallelTensor replica;
  // int fwd_input_idx = 0;
  // int bwd_input_idx = 0;
  // int fwd_output_idx = 0;
  // int bwd_output_idx = 0;
  // int init_output_idx = 0;
};

}; // namespace FlexFlow

#endif // _FLEXLOW_LINEAR_H

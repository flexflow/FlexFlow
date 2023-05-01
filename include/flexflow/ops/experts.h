#pragma once

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/ops/experts_params.h"

namespace FlexFlow {

class ExpertsMeta : public OpMeta {
public:
  ExpertsMeta(FFHandler handler,
              int _num_experts,
              int _experts_start_idx,
              int _data_dim,
              int _out_dim,
              int _experts_num_layers,
              int _experts_internal_dim_size,
              int _effective_batch_size,
              int _num_chosen_experts,
              float _alpha,
              bool _use_bias,
              ActiMode _activation);
  ~ExpertsMeta(void);

  // Thrust helper arrays
  int *sorted_indices;
  int *original_indices;
  int *non_zero_expert_labels;
  int *temp_sequence;
  int *exp_local_label_to_index;
  int *expert_start_indexes;
  int *num_assignments_per_expert; // numbers of tokes assigned to each expert.
                                   // Values may exceed the expert capacity
  int *capped_num_assignments_per_expert;
  int *destination_start_indices;
  float const **token_idx_array;
  float const **dev_weights;
  float const **weight_idx_array1;
  float const **weight_idx_array2;
  float const **coefficient_idx_array;
  float **output_idx_array;
  float const **bias_idx_array1;
  float const **bias_idx_array2;
  float const *one_ptr;
  float const **one_ptr_array;

  // array of arrays to store cublasGemmBatchedEx outputs before aggregation
  float **batch_outputs1;
  float **batch_outputs2;
  float **dev_batch_outputs1;
  float **dev_batch_outputs2;

  int num_experts;
  int experts_start_idx;
  int data_dim;
  int out_dim;
  int experts_num_layers;
  int experts_internal_dim_size;
  int effective_batch_size;
  int num_chosen_experts;
  int expert_capacity;
  float alpha;
  bool use_bias;
  ActiMode activation;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnActivationDescriptor_t actiDesc;
  cudnnTensorDescriptor_t resultTensorDesc1;
  cudnnTensorDescriptor_t resultTensorDesc2;
#else
  miopenActivationDescriptor_t actiDesc;
  miopenTensorDescriptor_t resultTensorDesc1;
  miopenTensorDescriptor_t resultTensorDesc2;
#endif
};

// definitions for the CUDA kernel
#define MAX_BATCH_SIZE 1024 * 2 // 32 * 10
#define MAX_EXPERTS_PER_BLOCK 32

class Experts : public Op {
public:
  using Params = ExpertsParams;
  using Input = std::vector<ParallelTensor>;
  Experts(FFModel &model,
          Params const &params,
          Input const &inputs,
          bool allocate_weights = false,
          char const *name = nullptr);
  Experts(FFModel &model,
          LayerID const &layer_guid,
          ParallelTensor const *inputs,
          int _num_experts,
          int _experts_start_idx,
          int _experts_output_dim_size,
          float _alpha,
          int _experts_num_layers,
          int _experts_internal_dim_size,
          bool _use_bias,
          ActiMode _activation,
          bool allocate_weights,
          char const *name = nullptr);
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

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
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               Input const &inputs,
                               int num_inputs);
  Params get_params() const;
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_kernel_wrapper(ExpertsMeta const *m,
                                     float const *input,
                                     int const *indices,
                                     float const *topk_gate_preds,
                                     float *output,
                                     float const *weights,
                                     float const *biases,
                                     int num_active_tokens,
                                     int chosen_experts,
                                     int batch_size,
                                     int out_dim);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

public:
  int num_experts;
  int experts_start_idx;
  int experts_output_dim_size;
  int data_dim;
  int out_dim;
  int effective_batch_size;
  int num_chosen_experts;
  float alpha;
  int experts_num_layers;
  int experts_internal_dim_size;
  bool use_bias;
  ActiMode activation;
};

}; // namespace FlexFlow

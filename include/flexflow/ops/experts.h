#pragma once

#include "flexflow/model.h"
#include "flexflow/ops/experts_params.h"

namespace FlexFlow {

class ExpertsMeta : public OpMeta {
public:
  ExpertsMeta(FFHandler handler,
              int _num_experts,
              int _experts_start_idx,
              float _alpha);
  ~ExpertsMeta(void);
  int num_experts;
  int experts_start_idx;
  float alpha;
  float **dev_region_ptrs;
};

// definitions for the CUDA kernel
#define MAX_BATCH_SIZE 32 * 10
#define MAX_EXPERTS_PER_BLOCK 32

class Experts : public Op {
public:
  using Params = ExpertsParams;
  using Input = std::vector<ParallelTensor>;
  Experts(FFModel &model,
          Params const &params,
          Input const &inputs,
          char const *name = nullptr);
  Experts(FFModel &model,
          ParallelTensor const *inputs,
          int _num_experts,
          int _experts_start_idx,
          int _experts_output_dim_size,
          float _alpha,
          int _experts_num_layers,
          int _experts_internal_dim_size,
          char const *name = nullptr);
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void inference(FFModel const &,
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
                                     float const *acc_input_ptr,
                                     int const *acc_indices_ptr,
                                     float const *acc_topk_gate_preds_ptr,
                                     float **outputs,
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
  float alpha;
  int experts_num_layers;
  int experts_internal_dim_size;
};

}; // namespace FlexFlow

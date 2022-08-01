#pragma once

#include "flexflow/model.h"

namespace FlexFlow {

class LayerNormMeta;

class LayerNorm : public Op {
public:
  LayerNorm(FFModel &model,
            const LayerID &_layer_guid,
            const ParallelTensor _input,
            const std::vector<int> &axes,
            bool _elementwise_affine,
            float _eps,
            bool allocate_weights,
            const char *name);
  void init(const FFModel &);
  void forward(const FFModel &);
  void backward(const FFModel &);
  void print_layer(const FFModel &model) { assert(0); }
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
  bool measure_operator_cost(Simulator *sim,
                             const MachineView &pc,
                             CostMetrics &cost_metrics) const;
  template <typename T>
  static void forward_kernel(const LayerNormMeta *m,
                             const T *input_ptr,
                             T *output_ptr,
                             T *gamma_ptr,
                             T *beta_ptr,
                             ffStream_t stream);
  template <typename T>
  static void forward_kernel_wrapper(const LayerNormMeta *m,
                                     const T *input_ptr,
                                     T *output_ptr,
                                     T *gamma_ptr,
                                     T *beta_ptr);
  template <typename T>
  static void backward_kernel(const LayerNormMeta *m,
                              const T *output_grad_ptr,
                              const T *input_ptr,
                              T *input_grad_ptr,
                              const T *gamma_ptr,
                              T *gamma_grad_ptr,
                              T *beta_grad_ptr,
                              ffStream_t stream);
  template <typename T>
  static void backward_kernel_wrapper(const LayerNormMeta *m,
                                      const T *output_grad_ptr,
                                      const T *input_ptr,
                                      T *input_grad_ptr,
                                      const T *gamma_ptr,
                                      T *gamma_grad_ptr,
                                      T *beta_grad_ptr);

public:
  bool elementwise_affine;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
};

class LayerNormMeta : public OpMeta {
public:
  LayerNormMeta(FFHandler handle, const LayerNorm *ln);

public:
  bool elementwise_affine;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  float *mean_ptr, *rstd_ptr, *ds_ptr, *db_ptr, *scale_ptr, *bias_ptr;
  char op_name[MAX_OPNAME];
};

}; // namespace FlexFlow

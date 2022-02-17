#ifndef _FLEXFLOW_SPLIT_H
#define _FLEXFLOW_SPLIT_H

#include "flexflow/model.h"

namespace FlexFlow {

class Split : public Op {
public:
  Split(FFModel& model,
        const ParallelTensor input,
        const std::vector<int>& split,
        int axis,
        const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void print_layer(const FFModel& model) override {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_kernel(float **out_ptrs,
                             float const *in_ptr,
                             Legion::coord_t const *out_blk_sizes,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t num_blks,
                             int numOutputs,
                             ffStream_t stream);
  static void forward_kernel_wrapper(float **out_ptrs,
                                     float const *in_ptr,
                                     Legion::coord_t const *out_blk_sizes,
                                     Legion::coord_t in_blk_size,
                                     Legion::coord_t num_blks,
                                     int numOutputs);
  static void backward_kernel(float *in_grad_ptr,
                              float const **out_grad_ptr,
                              Legion::coord_t const *out_blk_sizes,
                              Legion::coord_t in_blk_size,
                              Legion::coord_t num_blks,
                              int numOutputs,
                              ffStream_t stream);
  static void backward_kernel_wrapper(float *in_grad_ptr,
                                      float const **out_grad_ptr,
                                      Legion::coord_t const *out_blk_sizes,
                                      Legion::coord_t in_blk_size,
                                      Legion::coord_t num_blks,
                                      int numOutputs);
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const override;

  size_t get_params_hash() const override;
public:
  int axis;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_SPLIT_H

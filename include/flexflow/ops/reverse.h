#ifndef _FLEXFLOW_REVERSE_H_
#define _FLEXFLOW_REVERSE_H_

#include "flexflow/model.h"

namespace FlexFlow {

class Reverse : public Op {
public:
  Reverse(FFModel& model,
          const ParallelTensor input,
          int axis,
          const char* name);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void reset_idx(const FFModel&) override {assert(0);}
  void pipeinit(const FFModel&)  override {assert(0);}
  void pipeforward(const FFModel&)  override {assert(0);}
  void pipebackward(const FFModel&)  override {assert(0);}
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
  static void forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             Legion::coord_t num_out_blks,
                             Legion::coord_t reverse_dim_size,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t output_size,
			                       ffStream_t stream);
  static void forward_kernel_wrapper(float const *in_ptr,
                                     float *out_ptr,
                                     Legion::coord_t num_out_blks,
                                     Legion::coord_t reverse_dim_size,
                                     Legion::coord_t in_blk_size,
                                     Legion::coord_t output_size);
  static void backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              Legion::coord_t num_out_blks,
                              Legion::coord_t reverse_dim_size,
                              Legion::coord_t in_blk_size,
                              Legion::coord_t input_size,
			                        ffStream_t stream);
  static void backward_kernel_wrapper(float const *out_grad_ptr,
                                      float *in_grad_ptr,
                                      Legion::coord_t num_out_blks,
                                      Legion::coord_t reverse_dim_size,
                                      Legion::coord_t in_blk_size,
                                      Legion::coord_t input_size);
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const override;
public:
  int axis;
};

}; // namespace FlexFlow

#endif


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
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  static void forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             Legion::coord_t num_out_blks,
                             Legion::coord_t reverse_dim_size,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t output_size,
			                       cudaStream_t stream);
  static void backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              Legion::coord_t num_out_blks,
                              Legion::coord_t reverse_dim_size,
                              Legion::coord_t in_blk_size,
                              Legion::coord_t input_size,
			                        cudaStream_t stream);
#else
  static void forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             Legion::coord_t num_out_blks,
                             Legion::coord_t reverse_dim_size,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t output_size,
			                       hipStream_t stream);
  static void backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              Legion::coord_t num_out_blks,
                              Legion::coord_t reverse_dim_size,
                              Legion::coord_t in_blk_size,
                              Legion::coord_t input_size,
			                        hipStream_t stream);
#endif
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
public:
  int axis;
};

}; // namespace FlexFlow

#endif


#ifndef _FLEXFLOW_SPLIT_H
#define _FLEXFLOW_SPLIT_H

#include "flexflow/model.h"

namespace FlexFlow {

class Split : public Op {
public:
  Split(FFModel &model,
        const ParallelTensor input,
        std::vector<int> const &split,
        int legion_axis,
        char const *name);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  static OpMeta *init_task(Legion::Task const *task,
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
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  size_t get_params_hash() const override;

public:
  int legion_axis;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_SPLIT_H

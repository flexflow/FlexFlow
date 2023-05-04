#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_SIM_ENVIRONMENT_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_SIM_ENVIRONMENT_H

#include "kernels/accessor.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <vector>
#include "cost_metrics.h"

namespace FlexFlow {

struct SimEnvironment { };

struct Simulator {
  SimEnvironment new_environment() const;
};

GenericTensorAccessorW allocate_input(SimEnvironment &sim, TensorShape const &);
GenericTensorAccessorW allocate_input(SimEnvironment &sim, ParallelTensorShape const &);
std::vector<GenericTensorAccessorW> allocate_input(SimEnvironment &sim, std::vector<ParallelTensorShape> const &);

GenericTensorAccessorW allocate_weight(SimEnvironment &sim, TensorShape const &);
GenericTensorAccessorW allocate_weight(SimEnvironment &sim, ParallelTensorShape const &);
std::vector<GenericTensorAccessorW> allocate_weight(SimEnvironment &sim, std::vector<ParallelTensorShape> const &);

size_t get_input_memory_usage(SimEnvironment const &);
size_t get_output_memory_usage(SimEnvironment const &);
size_t get_weights_memory_usage(SimEnvironment const &);
size_t get_op_total_memory(SimEnvironment const &);

CostMetrics make_metrics(float forward_time, float backward_time, float sync_time, SimEnvironment const &);

float default_estimate_sync_time(SimEnvironment const &);

}

#endif

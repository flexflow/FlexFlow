#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_BACKING_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_BACKING_H

#include "pcg/computation_graph.h"
#include "kernels/allocation.h"
#include "op-attrs/operator_attrs.h"
#include "op_task_signature.h"
#include "task_argument_accessor.h"
#include <vector>
#include <unordered_map>
#include <functional>

namespace FlexFlow {

using TaskImplFunction = std::function<optional<float>(TaskArgumentAccessor const &)>;

struct LocalTrainingBacking {
  LocalTrainingBacking(ComputationGraph, Allocator, Tensor, Tensor);
  ~LocalTrainingBacking() = default;

  GenericTensorAccessorR execute_forward();
  void execute_backward();
  void execute_update();

  LocalTaskArgumentAccessor get_fwd_accessor(OpTaskInvocation);
  LocalTaskArgumentAccessor get_bwd_accessor(OpTaskInvocation);

private:
  ComputationGraph computation_graph;
  Allocator allocator;
  std::vector<Node> topologically_ordered_graph;
  GenericTensorAccessorR input_tensor_backing;
  GenericTensorAccessorR output_tensor_backing;

  // hold mappings
  std::unordered_map<task_id_t, void*> task_id_impl_mapping;
  std::unordered_map<task_id_t, OpTaskSignature> task_id_signature_mapping;
};

} // namespace FlexFlow

#endif
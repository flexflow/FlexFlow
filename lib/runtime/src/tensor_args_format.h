#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_ARGS_FORMAT_H

#include "executable_task_invocation.h"
#include "legion.h"
#include "parallel_computation_graph.h"
#include "parallel_tensor_guid_t.h"
#include "task_argument_accessor.h"
#include "task_signature.h"

namespace FlexFlow {

struct TensorArgsFormat {
  bidict<parallel_tensor_guid_t, region_idx_t> region_idxs;
  std::unordered_map<parallel_tensor_guid_t, Permissions> privs_map;
  std::unordered_map<parallel_tensor_guid_t, DataType> datatypes;
  std::unordered_map<slot_id, parallel_tensor_guid_t>
      nonvariadic_slot_to_tensor;
  std::unordered_map<slot_id, std::vector<parallel_tensor_guid_t>>
      variadic_slot_to_tensor;
};

void add_tensor_requirements(Legion::TaskLauncher const &,
                             TensorArgsFormat const &);
void add_tensor_requirements(Legion::IndexTaskLauncher const &,
                             TensorArgsFormat const &);

bool includes_tensor(ExecutableTensorSpec const &,
                     parallel_tensor_guid_t const &);
std::unordered_set<slot_id> get_tensor_slots(ExecutableTaskBinding const &,
                                             parallel_tensor_guid_t const &);
Permissions get_tensor_permissions(TaskSignature const &,
                                   ExecutableTaskBinding const &,
                                   parallel_tensor_guid_t const &);
std::vector<parallel_tensor_guid_t> as_vector(ExecutableTensorSpec const &);

TensorArgsFormat process_tensor_args(TaskSignature const &,
                                     ParallelComputationGraph const &,
                                     ExecutableTaskBinding const &);

} // namespace FlexFlow

#endif

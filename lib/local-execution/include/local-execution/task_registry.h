
#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/task_signature_impl.h"
#include "op-attrs/operator_attrs.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

struct TaskRegistry {
  TaskRegistry() = default;

  void register_tasks_for_layer(layer_guid_t const &,
                                ComputationGraphOpAttrs const &attrs);

  std::unordered_map<layer_guid_t, std::optional<task_id_t>> init_task_ids;
  std::unordered_map<layer_guid_t, std::optional<task_id_t>> forward_task_ids;
  std::unordered_map<layer_guid_t, std::optional<task_id_t>> backward_task_ids;
  std::unordered_map<task_id_t, TaskSignatureAndImpl> task_mapping;

  bool operator==(TaskRegistry const &other) const;
  bool operator!=(TaskRegistry const &other) const;

private:
  std::tuple<decltype(init_task_ids) const &,
             decltype(forward_task_ids) const &,
             decltype(backward_task_ids) const &,
             decltype(task_mapping) const &>
      tie() const;
};

std::string format_as(
    std::unordered_map<layer_guid_t, std::optional<task_id_t>> const &x);
std::ostream &operator<<(
    std::ostream &s,
    std::unordered_map<layer_guid_t, std::optional<task_id_t>> const &x);

std::string
    format_as(std::unordered_map<task_id_t, TaskSignatureAndImpl> const &x);
std::ostream &
    operator<<(std::ostream &s,
               std::unordered_map<task_id_t, TaskSignatureAndImpl> const &x);

std::string format_as(TaskRegistry const &x);
std::ostream &operator<<(std::ostream &s, TaskRegistry const &x);

} // namespace FlexFlow

#endif

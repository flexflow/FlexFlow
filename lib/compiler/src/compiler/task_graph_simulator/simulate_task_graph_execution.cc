#include "compiler/task_graph_simulator/simulate_task_graph_execution.h"
#include "compiler/task_graph_simulator/in_progress_task.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_state.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_trace.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/set_of.h"
#include "utils/containers/sorted.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/overload.h"
#include <functional>
#include <set>

namespace FlexFlow {

TaskGraphExecutionTrace simulate_task_graph_execution(
    DiGraphView const &task_graph,
    std::function<float(Node const &)> cost_function,
    TaskExecutionConstraint const &constraint) {
  if (!is_acyclic(task_graph)) {
    PANIC(
        "simulate_task_graph_execution cannot simulate cyclic directed graphs");
  }

  TaskGraphExecutionState execution_state = TaskGraphExecutionState{
      /*ready_tasks=*/set_of(get_initial_nodes(task_graph)),
      /*in_progress_tasks=*/{},
      /*finished_tasks=*/{},
      /*current_time=*/0.0};

  std::set<TaskProfile> task_profiles;

  auto start_task_processing = [&](Node const &task) {
    float cost = cost_function(task);
    execution_state.in_progress_tasks.push(
        InProgressTask{execution_state.current_time,
                       execution_state.current_time + cost,
                       task});
    execution_state.ready_tasks.erase(task);
  };

  auto dependencies_are_satisfied = [&](Node const &task) {
    std::set<Node> incoming_dependencies = get_predecessors(task_graph, task);
    return is_subseteq_of(incoming_dependencies,
                          execution_state.finished_tasks);
  };

  auto finish_task_processing = [&](InProgressTask const &in_progress_task) {
    execution_state.finished_tasks.insert(in_progress_task.node);
    for (Node const &task : get_successors(task_graph, in_progress_task.node)) {
      if (dependencies_are_satisfied(task)) {
        execution_state.ready_tasks.insert(task);
      }
    }
    task_profiles.insert(TaskProfile{in_progress_task.node,
                                     in_progress_task.start_time,
                                     in_progress_task.end_time});
    execution_state.current_time = in_progress_task.end_time;
  };

  auto is_processing_done = [&]() {
    return execution_state.ready_tasks.empty() &&
           execution_state.in_progress_tasks.empty();
  };

  auto get_next_task_to_finish = [&]() {
    InProgressTask task = execution_state.in_progress_tasks.top();
    execution_state.in_progress_tasks.pop();
    return task;
  };

  while (!is_processing_done()) {
    auto ready_tasks_copy = execution_state.ready_tasks;
    for (Node const &task : ready_tasks_copy) {
      std::set<Node> raw_in_progress_tasks =
          transform(set_of(execution_state.in_progress_tasks.contents()),
                    [](InProgressTask const &t) { return t.node; });

      if (constraint.is_satisfied(
              task, raw_in_progress_tasks, execution_state.finished_tasks)) {
        start_task_processing(task);
      }
    }

    if (!execution_state.in_progress_tasks.empty()) {
      InProgressTask next_task = get_next_task_to_finish();
      finish_task_processing(next_task);
    } else {
      PANIC("Constraints cannot be satisfied");
    }
  }
  if (execution_state.finished_tasks.size() != num_nodes(task_graph)) {
    PANIC("Failed to execute all tasks in given graph");
  }

  return TaskGraphExecutionTrace{task_profiles};
}

} // namespace FlexFlow

#include "compiler/task_graph_simulator/task_graph_execution_trace.h"
#include "utils/containers/maximum.h"
#include "utils/containers/minimum.h"
#include "utils/containers/transform.h"
#include "utils/fmt/set.h"

namespace FlexFlow {

float get_total_execution_time(TaskGraphExecutionTrace const &trace) {
  ASSERT(!trace.task_profiles.empty());

  float end_time =
      maximum(transform(trace.task_profiles, [](TaskProfile const &profile) {
        return profile.end_time;
      }));
  float start_time =
      minimum(transform(trace.task_profiles, [](TaskProfile const &profile) {
        return profile.start_time;
      }));

  return end_time - start_time;
}

} // namespace FlexFlow

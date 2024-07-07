#include "substitutions/unlabelled/pattern_value.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenDataflowValue raw_dataflow_value_from_pattern_value(PatternValue const &v) {
  return v.visit<OpenDataflowValue>(overload {
    [](PatternNodeOutput const &o) { return o.raw_dataflow_output; },
    [](PatternInput const &i) { return i.raw_dataflow_graph_input; },
  });
}

} // namespace FlexFlow

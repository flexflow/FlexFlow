#include "utils/graph/open_multidigraph/open_multi_di_edge.h"

namespace FlexFlow {

bool is_input_edge(OpenMultiDiEdge const &e) {
  return std::holds_alternative<InputMultiDiEdge>(e);
}

bool is_output_edge(OpenMultiDiEdge const &e) {
  return std::holds_alternative<OutputMultiDiEdge>(e);
}

bool is_standard_edge(OpenMultiDiEdge const &e) {
  return std::holds_alternative<MultiDiEdge>(e);
}


} // namespace FlexFlow

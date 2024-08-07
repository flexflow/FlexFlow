#include "compiler/graph_optimize_state.h"

namespace FlexFlow {

bool GraphOptimizeState::operator==(GraphOptimizeState const &other) const {
  auto layers1 = topological_ordering(graph_optimize_result.pcg);
  auto layers2 = topological_ordering(other.graph_optimize_result.pcg);
  if (layers1.size() != layers2.size()) {
    return false;
  }
  for (size_t i = 0; i < layers1.size(); ++i) {
    auto inputs1 = get_layer_inputs(graph_optimize_result.pcg, layers1[i]);
    auto inputs2 =
        get_layer_inputs(other.graph_optimize_result.pcg, layers2[i]);
    if (inputs1.size() != inputs2.size()) {
      return false;
    }
    for (size_t j = 0; j < inputs1.size(); ++j) {
      if (inputs1[j] != inputs2[j]) {
        return false;
      }
    }
  }
  return true;
}

bool GraphOptimizeState::operator!=(GraphOptimizeState const &other) const {
  return !(*this == other);
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::GraphOptimizeState>::operator()(
    ::FlexFlow::GraphOptimizeState const &state) const {
  size_t seed = 0;
  auto layers = topological_ordering(state.graph_optimize_result.pcg);
  ::FlexFlow::hash_combine(seed, layers.size());
  for (auto layer : layers) {
    auto inputs = get_layer_inputs(state.graph_optimize_result.pcg, layer);
    ::FlexFlow::hash_combine(seed, inputs.size());
    for (auto input : inputs) {
      ::FlexFlow::hash_combine(seed, input);
    }
  }
  return seed;
}

} // namespace std

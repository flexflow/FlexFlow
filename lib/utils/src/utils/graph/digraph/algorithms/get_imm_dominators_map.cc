#include "utils/graph/digraph/algorithms/get_imm_dominators_map.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_element_counts.h"
#include "utils/containers/keys.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/containers/filter_values.h"
#include "utils/graph/digraph/algorithms/get_dominators_map.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_map<Node, std::optional<Node>>
    get_imm_dominators_map(DiGraphView const &g) {

  std::unordered_map<Node, std::unordered_set<Node>> node_to_its_dominators = get_dominators_map(g);

  auto get_imm_dominator = [&](Node const &n) {
    std::unordered_set<Node> n_dominators = node_to_its_dominators.at(n);
    n_dominators.erase(n);
    std::vector<Node> recursive_dominator_list = concat_vectors(transform(as_vector(n_dominators), 
                                                                          [&](Node const &dominator) { return as_vector(node_to_its_dominators.at(dominator)); }));
    std::unordered_map<Node, int> dominator_counts = get_element_counts(recursive_dominator_list);
    std::unordered_set<Node> imm_dominators = keys(filter_values(dominator_counts, [](int count) { return count <= 1; }));
    assert (imm_dominators.size() <= 1);

    return maybe_get_only(imm_dominators);
  };

  return generate_map(get_nodes(g), get_imm_dominator);
}

} // namespace FlexFlow

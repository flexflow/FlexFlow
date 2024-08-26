#include "substitutions/pcg_pattern_match.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/bidict/algorithms/merge_bidicts.h"
#include "utils/containers/zip.h"

namespace FlexFlow {

bidict<PatternNodeOutput, parallel_tensor_guid_t>
    get_output_mapping_for_pcg_pattern_match(
        PCGPatternMatch const &match,
        PCGPattern const &pattern,
        SubParallelComputationGraph const &spcg) {
  bidict<PatternNodeOutput, parallel_tensor_guid_t> result;

  for (auto const &[pattern_node, matched_layer] : match.node_assignment) {
    std::vector<parallel_tensor_guid_t> matched_layer_output_tensors =
        get_layer_outputs(spcg, matched_layer);
    std::vector<PatternNodeOutput> pattern_node_outputs =
        get_pattern_node_outputs(pattern, pattern_node);

    assert(matched_layer_output_tensors.size() == pattern_node_outputs.size());

    bidict<PatternNodeOutput, parallel_tensor_guid_t> mapping =
        bidict_from_keys_and_values(pattern_node_outputs,
                                    matched_layer_output_tensors);

    result = merge_bidicts(result, mapping);
  }

  return result;
}

} // namespace FlexFlow

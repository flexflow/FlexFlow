#include "substitutions/pcg_pattern_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"

namespace FlexFlow {

PCGPatternBuilder::PCGPatternBuilder() 
  : g(LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>
    ::create<UnorderedSetLabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>>())
{ }

// PatternValue add_input() {
//   return tensor_attribute_pattern_match_all();
// }
//
// PatternValue PCGPatternBuilder::add_input(TensorAttributePattern const &p) {
//   return PatternValue{PatternInput{this->g.add_input(p)}};
// }
//
// std::vector<PatternValue> PCGPatternBuilder::add_operator(OperatorAttributePattern const &p,
//                                              std::vector<PatternValue> const &inputs,
//                                              std::vector<TensorAttributePattern> const &outputs) {
//   NodeAddedResult node_added_result = this->g.add_node(p, 
//                                                        transform(inputs, raw_open_dataflow_value_from_pattern_value),
//                                                        outputs);
//   return transform(node_added_result.outputs, pattern_value_from_raw_open_dataflow_value);
// }
//
// PatternValue PCGPatternBuilder::add_operator(OperatorAttributePattern const &p,
//                                              std::vector<PatternValue> const &inputs,
//                                              TensorAttributePattern const &output) {
//   return get_only(this->add_operator(p, inputs, {output}));
// }
//
//
// PCGPattern PCGPatternBuilder::get_pattern() const {
//   return PCGPattern{this->g};
// }

} // namespace FlexFlow

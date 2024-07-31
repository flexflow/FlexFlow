#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_OPERATOR_ATTRS_ASSIGNMENT_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_OPERATOR_ATTRS_ASSIGNMENT_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.dtg.h"

namespace FlexFlow {

OutputOperatorAttrsAssignment output_operator_clone_node(PatternNode const &);

PCGOperatorAttrs materialize_output_operator_from_attrs_assignment(OutputOperatorAttrsAssignment const &attrs_assignment,
                                                                   std::unordered_map<PatternNode, PCGOperatorAttrs> const &node_match);

} // namespace FlexFlow

#endif

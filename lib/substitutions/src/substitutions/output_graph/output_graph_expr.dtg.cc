// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/output_graph/output_graph_expr.struct.toml
/* proj-data
{
  "generated_from": "9084c9afb2724504a6f4db4288a83a0d"
}
*/

#include "substitutions/output_graph/output_graph_expr.dtg.h"

namespace FlexFlow {
OutputGraphExpr::OutputGraphExpr(
    ::FlexFlow::NodeLabelledOpenMultiDiGraph<
        ::FlexFlow::OutputOperatorAttrsAssignment> const &raw_graph)
    : raw_graph(raw_graph) {}
} // namespace FlexFlow

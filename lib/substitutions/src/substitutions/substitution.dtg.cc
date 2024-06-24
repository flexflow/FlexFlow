// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/substitution.struct.toml
/* proj-data
{
  "generated_from": "9e0ea4f3e23858068cc975534e6c4cf7"
}
*/

#include "substitutions/substitution.dtg.h"

namespace FlexFlow {
Substitution::Substitution(
    ::FlexFlow::PCGPattern const &pcg_pattern,
    ::FlexFlow::OutputGraphExpr const &output_graph_expr,
    ::FlexFlow::bidict<::FlexFlow::DataflowGraphInput,
                       ::FlexFlow::OpenDataflowValue> const
        &input_edge_match_to_output,
    ::FlexFlow::bidict<::FlexFlow::DataflowOutput,
                       ::FlexFlow::DataflowOutput> const
        &output_edge_match_to_output)
    : pcg_pattern(pcg_pattern), output_graph_expr(output_graph_expr),
      input_edge_match_to_output(input_edge_match_to_output),
      output_edge_match_to_output(output_edge_match_to_output) {}
} // namespace FlexFlow

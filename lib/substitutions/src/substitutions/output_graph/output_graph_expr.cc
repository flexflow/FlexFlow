#include "substitutions/output_graph/output_graph_expr.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"

namespace FlexFlow {

std::vector<OutputGraphExprNodeOutput>
    get_node_outputs(OutputGraphExpr const &g, OutputGraphExprNode const &n) {
  std::vector<DataflowOutput> raw_outputs =
      get_outputs(g.raw_graph, n.raw_graph_node);

  return transform(raw_outputs, [](DataflowOutput const &o) {
    return OutputGraphExprNodeOutput{o};
  });
}

} // namespace FlexFlow

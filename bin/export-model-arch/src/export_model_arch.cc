#include "pcg/computation_graph/computation_graph_edge.h"
#include "pcg/file_format/v1/v1_computation_graph.h"
#include "utils/cli/cli_spec.h"
#include "utils/exception.h"
#include "utils/cli/cli_parse_result.h"
#include "models/transformer.h"
#include "utils/graph/digraph/algorithms/digraph_as_dot.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/serial_parallel/get_serial_parallel_decomposition.h"
#include "pcg/computation_graph.h"
#include "op-attrs/computation_graph_op_attrs.h"

using namespace ::FlexFlow;

SerialParallelDecomposition get_computation_graph_serial_parallel_decomposition(ComputationGraph const &cg) {
  std::unordered_set<layer_guid_t> weight_and_input_layers = filter(get_layers(cg), [&](layer_guid_t const &l) {  
    ComputationGraphOpAttrs op_attrs = get_layer_attrs(cg, l).attrs;
    return op_attrs.has<WeightAttrs>() || op_attrs.has<InputAttrs>();
  });

  std::unordered_set<layer_guid_t> weight_and_input_layer_successors = transform(
      get_subgraph_outgoing_edges(cg, weight_and_input_layers),
      get_computation_graph_edge_dst_layer);

  // DiGraphView preprocessed_digraph = [&] {
  DiGraph digraph = materialize_digraph_view<AdjacencyDiGraph>(cg.raw_graph);
  // Node fake_node = digraph.add_node();
  for (layer_guid_t const &src : weight_and_input_layers) {
    for (layer_guid_t const &dst : weight_and_input_layer_successors) {
      digraph.add_edge(DirectedEdge{src.raw_node, dst.raw_node});
    }
  }
  DiGraphView preprocessed_digraph = digraph;
    // return digraph;
  // }();

  // std::function<std::string(Node const &)> get_node_label =
  //     [&](Node const &n) -> std::string {
  //   if (n == fake_node) {
  //     return "FAKE";
  //   }
  //   LayerAttrs a = cg.raw_graph.at(n);
  //   RecordFormatter r = as_dot(a.attrs);
  //
  //
  //   if (a.name.has_value()) {
  //     RecordFormatter rr;
  //     rr << "Name" << a.name.value();
  //     r << rr;
  //   }
  //
  //   std::ostringstream oss;
  //   oss << r;
  //   return oss.str();
  // };
  // std::string preprocessed_dot = digraph_as_dot(transitive_reduction(preprocessed_digraph), get_node_label);
  // std::cout << preprocessed_dot << std::endl;
  // exit(0);

  SerialParallelDecomposition sp_decomposition = get_serial_parallel_decomposition(preprocessed_digraph).value();
  
  return sp_decomposition;
}

BinarySPDecompositionTree get_computation_graph_right_assoc_sp_decomposition(ComputationGraph const &cg) {
  return right_associative_binary_sp_tree_from_nary(get_computation_graph_serial_parallel_decomposition(cg));
}

ComputationGraph get_default_transformer_computation_graph() {
  TransformerConfig config = get_default_transformer_config();
  ComputationGraph cg = get_transformer_computation_graph(config);
  
  return cg;
}

ComputationGraph get_model_computation_graph(std::string const &model_name) {
  if (model_name == "transformer") {
    return get_default_transformer_computation_graph();
  } else {
    throw mk_runtime_error(fmt::format("Unknown model name: {}", model_name));
  }
}

int main(int argc, char **argv) {
  CLISpec cli = empty_cli_spec();
  CLIArgumentKey key_sp_decomposition = cli_add_flag(cli, CLIFlagSpec{"--sp-decomposition", std::nullopt});
  std::unordered_set<std::string> model_options = {"transformer"};
  CLIArgumentKey key_model_name = cli_add_positional_argument(cli, CLIPositionalArgumentSpec{model_options});

  CLIParseResult parsed = ({
    tl::expected<CLIParseResult, std::string> result = cli_parse(cli, argc, argv);
    if (!result.has_value()) {
      throw mk_runtime_error(result.error());
    }

    result.value();
  });

  std::string model_name = cli_get_argument(parsed, key_model_name);
  bool sp_decompositition = cli_get_flag(parsed, key_sp_decomposition);

  if (sp_decompositition) {
    throw mk_runtime_error("Exporting sp decomposition is currently unsupported.");
  }

  ComputationGraph cg = get_model_computation_graph(model_name);
  BinarySPDecompositionTree sp_decomposition = get_computation_graph_right_assoc_sp_decomposition(cg);


  // std::cout << as_dot(cg) << std::endl;
  // nlohmann::json j = to_v1(cg);
  nlohmann::json j = sp_decomposition;

  std::cout << j.dump(2) << std::endl;

  return 0;
}

#include "utils/graph/serial_parallel/serialparallel_internal.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/serial_parallel/sink_settings.dtg.h" 
#include "utils/graph/serial_parallel/source_settings.dtg.h"
#include "utils/graph/serial_parallel/split_ast_node.dtg.h"

namespace FlexFlow {

std::unordered_set<Node>
    from_source_to_sink(DiGraphView const &g,
                        std::unordered_set<Node> const &srcs,
                        std::unordered_set<Node> const &sinks,
                        SourceSettings include_src,
                        SinkSettings include_sink) {
  assert(is_acyclic(g));

  Node contracted_src = get_first(srcs);
  Node contracted_sink = get_first(sinks);
  std::unordered_map<Node, Node> contraction;
  for (Node const &src : srcs) {
    contraction.insert({src, contracted_src});
  }
  for (Node const &sink : sinks) {
    contraction.insert({sink, contracted_sink});
  }
  auto contracted_view = apply_contraction(g, contraction);

  std::unordered_set<Node> result =
      from_source_to_sink(contracted_view, contracted_src, contracted_sink);
  result.erase(contracted_src);
  result.erase(contracted_sink);

  if (include_src == SourceSettings::INCLUDE_SOURCE_NODES) {
    result = set_union(result, srcs);
  }
  if (include_sink == SinkSettings::INCLUDE_SINK_NODES) {
    result = set_union(result, sinks);
  }
  return result;
}

DiGraphView source_to_sink_subgraph(DiGraphView const &g,
                                    std::unordered_set<Node> const &srcs,
                                    std::unordered_set<Node> const &sinks,
                                    SourceSettings include_src,
                                    SinkSettings include_sink) {
  return get_subgraph(
      g, from_source_to_sink(g, srcs, sinks, include_src, include_sink));
}

IntermediateSpDecompositionTree sp_decomposition(DiGraphView const &g) {
  if (num_nodes(g) == 1) {
    return get_only(get_nodes(g));
  }

  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sinks(g);

  std::optional<Node> bottleneck = find_bottleneck_node(g);
  if (bottleneck.has_value()) {
    return IntermediateSpDecompositionTree{
      SplitType::SERIAL,
        {
          sp_decomposition(source_to_sink_subgraph(g,
                                                   sources,
                                                   {bottleneck.value()},
                                                   SourceSettings::INCLUDE_SOURCE_NODES,
                                                   SinkSettings::EXCLUDE_SINK_NODES)),
          sp_decomposition(source_to_sink_subgraph(g,
                                                   {bottleneck.value()},
                                                   sinks,
                                                   SourceSettings::INCLUDE_SOURCE_NODES,
                                                   SinkSettings::INCLUDE_SINK_NODES))}};
  } else {
    return parallel_decomposition(g);
  }
}

IntermediateSpDecompositionTree parallel_decomposition(DiGraphView const &g) {
  std::unordered_set<std::unordered_set<Node>> weakly_connected_components =
      get_weakly_connected_components(g);
  assert(weakly_connected_components.size() > 1);

  IntermediateSpDecompositionTree split(SplitType::PARALLEL, {});
  for (auto const &component : weakly_connected_components) {
    split.children.push_back(sp_decomposition(get_subgraph(g, component)));
  }

  return split;
}

SplitAST flatten_ast(SplitAST const &ast) {
  return visit(FlattenAST{}, ast);
}

struct ToFinalAST {
  std::variant<Serial, Parallel, Node> operator()(IntermediateSpDecompositionTree const &node) {
    if (node.type == SplitType::SERIAL) {
      return Serial{transform(node.children, [](SplitAST const &s) {
        return narrow<std::variant<Parallel, Node>>(internal_to_final_ast(s)).value();
      })};
    } else {
      return Parallel{transform(node.children, [](SplitAST const &s) {
        return narrow<std::variant<Serial, Node>>(internal_to_final_ast(s)).value();
      })};
    }
  }

  std::variant<Serial, Parallel, Node> operator()(Node const &node) {
    return node;
  }
};

std::variant<Serial, Parallel, Node> internal_to_final_ast(SplitAST const &ast) {
  return visit(ToFinalAST{}, ast);
}

SerialParallelDecomposition to_final_ast(SplitAST const &ast) {
  return std::visit([](auto &&x) { return SerialParallelDecomposition{x}; },
                    internal_to_final_ast(ast));
}

} // namespace FlexFlow

#include "utils/graph/serial_parallel/serialparallel_internal.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/sink_settings.dtg.h"
#include "utils/graph/serial_parallel/source_settings.dtg.h"
#include "utils/containers/extend.h"
#include "utils/containers/transform.h"
#include "utils/containers/get_only.h"
#include "utils/containers/get_first.h"

namespace FlexFlow {

Node find_source_node(DiGraphView const &g) {
  std::unordered_set<Node> srcs = get_sources(g);
  return get_only(srcs);
}

Node find_sink_node(DiGraphView const &g) {
  std::unordered_set<Node> sinks = get_sinks(g);
  return get_only(sinks);
}

std::optional<Node> find_bottleneck_node(DiGraphView const &g) {
  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sinks(g);

  std::optional<Node> maybe_bottleneck = get_imm_post_dominator(g, sources);
  if (maybe_bottleneck.has_value()) {
    assert(contains(get_dominators(g, sinks), maybe_bottleneck.value()));
  }
  return maybe_bottleneck;
}

std::unordered_set<Node> from_source_to_sink(DiGraphView const &g,
                                             Node const &src,
                                             Node const &sink) {
  assert(contains(get_dominators(g, sink), src));

  std::vector<Node> bfs = get_bfs_ordering(g, {src});
  auto end = find(bfs, sink);
  assert(end != bfs.end());

  std::unordered_set<Node> result(bfs.cbegin(), ++end);
  return result;
}

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

std::variant<IntermediateSpDecompositionTree, Node>
    sp_decomposition(DiGraphView const &g) {
  if (num_nodes(g) == 1) {
    return get_only(get_nodes(g));
  }

  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sinks(g);

  std::optional<Node> bottleneck = find_bottleneck_node(g);
  if (bottleneck.has_value()) {
    return IntermediateSpDecompositionTree{
        SplitType::SERIAL,
        {sp_decomposition(
             source_to_sink_subgraph(g,
                                     sources,
                                     {bottleneck.value()},
                                     SourceSettings::INCLUDE_SOURCE_NODES,
                                     SinkSettings::EXCLUDE_SINK_NODES)),
         sp_decomposition(
             source_to_sink_subgraph(g,
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

struct FlattenAST {
  void add_flattened_child_to_parent(
      IntermediateSpDecompositionTree &parent,
      std::variant<IntermediateSpDecompositionTree, Node> const &child) {
    if (std::holds_alternative<Node>(child)) {
      parent.children.push_back(child);
      return;
    }

    IntermediateSpDecompositionTree child_node =
        get<IntermediateSpDecompositionTree>(child);

    if (parent.type == child_node.type) {
      extend(parent.children, child_node.children);
    } else {
      parent.children.push_back(child);
    }
  }

  std::variant<IntermediateSpDecompositionTree, Node>
      operator()(IntermediateSpDecompositionTree const &ast_node) {
    IntermediateSpDecompositionTree result(ast_node.type, {});
    for (std::variant<IntermediateSpDecompositionTree, Node> const &child :
         ast_node.children) {
      std::variant<IntermediateSpDecompositionTree, Node> flattened_child =
          flatten_ast(child);
      add_flattened_child_to_parent(result, flattened_child);
    }
    return result;
  }

  std::variant<IntermediateSpDecompositionTree, Node>
      operator()(Node const &ast_node) {
    return ast_node;
  }
};

std::variant<IntermediateSpDecompositionTree, Node> flatten_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit(FlattenAST{}, ast);
}

struct ToFinalAST {
  std::variant<SerialSplit, ParallelSplit, Node>
      operator()(IntermediateSpDecompositionTree const &node) {
    if (node.type == SplitType::SERIAL) {
      return SerialSplit{transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<ParallelSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          })};
    } else {
      return ParallelSplit{without_order(transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<SerialSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          }))};
    }
  }

  std::variant<SerialSplit, ParallelSplit, Node> operator()(Node const &node) {
    return node;
  }
};

std::variant<SerialSplit, ParallelSplit, Node> internal_to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit(ToFinalAST{}, flatten_ast(ast));
}

SerialParallelDecomposition to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit([](auto &&x) { return SerialParallelDecomposition{x}; },
                    internal_to_final_ast(ast));
}

} // namespace FlexFlow

#include "utils/graph/serialparallel.h"
#include "utils/graph/conversions.h"
#include "utils/containers.h"
#include "utils/graph/algorithms.h"
#include "serialparallel_internal.h"
#include "utils/graph/digraph.h"

namespace FlexFlow {

Node find_source_node(IDiGraphView const &g) {
  std::unordered_set<Node> srcs = get_sources(g);
  return get_only(srcs);
}

Node find_sink_node(IDiGraphView const &g) {
  std::unordered_set<Node> sinks = get_sinks(g);
  return get_only(sinks);
}

optional<Node> find_bottleneck_node(IMultiDiGraphView const &g) {
  return find_bottleneck_node(*unsafe_view_as_digraph(g));
}

optional<Node> find_bottleneck_node(IDiGraphView const &g) {
  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sources(g);

  optional<Node> maybe_bottleneck = get_imm_post_dominator(g, sources);
  if (maybe_bottleneck.has_value()) {
    assert (contains(get_dominators(g, sinks), maybe_bottleneck.value()));
  }
  return maybe_bottleneck;
}

enum class SourceSettings {
  INCLUDE_SOURCE_NODES, EXCLUDE_SOURCE_NODES
};

enum class SinkSettings {
  INCLUDE_SINK_NODES, EXCLUDE_SINK_NODES
};

std::unordered_set<Node> from_source_to_sink(IDiGraphView const &g, Node const &src, Node const &sink) {
  assert (contains(get_dominators(g, sink), src));

  std::vector<Node> bfs = get_bfs_ordering(g, {src});
  auto end = find(bfs, sink);
  assert (end != bfs.end());

  std::unordered_set<Node> result(bfs.cbegin(), ++end);
  return result;
}

std::unordered_set<Node> from_source_to_sink(IDiGraphView const &g, std::unordered_set<Node> const &srcs, std::unordered_set<Node> const &sinks, SourceSettings include_src, SinkSettings include_sink) {
  assert (is_acyclic(g));

  Node contracted_src = get_first(srcs);
  Node contracted_sink = get_first(sinks);
  std::unordered_map<Node, Node> contraction;
  for (Node const &src : srcs) {
    contraction.insert({src, contracted_src});
  }
  for (Node const &sink : sinks) {
    contraction.insert({sink, contracted_sink});
  }
  auto contracted_view = unsafe_view_as_contracted(g, contraction);
  
  std::unordered_set<Node> result = from_source_to_sink(*contracted_view, contracted_src, contracted_sink);
  result.erase(contracted_src);
  result.erase(contracted_sink);

  if (include_src == SourceSettings::INCLUDE_SOURCE_NODES) {
    result = set_union(result, srcs);
  }
  if (include_sink == SinkSettings::EXCLUDE_SINK_NODES) {
    result = set_union(result, sinks);
  }
  return result;
}

std::unique_ptr<IDiGraphView> unsafe_source_to_sink_subgraph(IDiGraphView const &g, 
                                              std::unordered_set<Node> const &srcs, 
                                              std::unordered_set<Node> const &sinks, 
                                              SourceSettings include_src, 
                                              SinkSettings include_sink) {
  return unsafe_view_subgraph(g, from_source_to_sink(g, srcs, sinks, include_src, include_sink));
}

SplitAST sp_decomposition(IDiGraphView const &g) {
  if (num_nodes(g) == 1) {
    return get_only(get_nodes(g));
  }

  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sinks(g);

  optional<Node> bottleneck = find_bottleneck_node(g);
  if (bottleneck.has_value()) {
    return SplitASTNode(
      SplitType::SERIAL,
      sp_decomposition(*unsafe_source_to_sink_subgraph(g, 
                                                       sources, 
                                                       {bottleneck.value()}, 
                                                       SourceSettings::INCLUDE_SOURCE_NODES, 
                                                       SinkSettings::INCLUDE_SINK_NODES)),
      sp_decomposition(*unsafe_source_to_sink_subgraph(g, 
                                                       {bottleneck.value()}, 
                                                       sinks, 
                                                       SourceSettings::EXCLUDE_SOURCE_NODES, 
                                                       SinkSettings::INCLUDE_SINK_NODES))
    );
  } else {
    return parallel_decomposition(g);
  }
}

SplitAST parallel_decomposition(IDiGraphView const &g) {
  std::vector<std::unordered_set<Node>> weakly_connected_components = get_weakly_connected_components(g);
  assert (weakly_connected_components.size() > 1);

  SplitASTNode split(SplitType::PARALLEL);
  for (auto const &component : weakly_connected_components) {
    split.children.push_back(sp_decomposition(*unsafe_view_subgraph(g, component)));
  }

  return split;
}

struct FlattenAST {
  void add_flattened_child_to_parent(SplitASTNode &parent, SplitAST const &child) {
    if (mpark::holds_alternative<Node>(child)) {
      parent.children.push_back(child);
      return;
    }

    SplitASTNode child_node = mpark::get<SplitASTNode>(child);

    if (parent.type == child_node.type) {
      extend(parent.children, child_node.children);
    } else {
      parent.children.push_back(child); 
    }
  }

  SplitAST operator()(SplitASTNode const &ast_node) {
    SplitASTNode result(ast_node.type);
    for (SplitAST const &child : ast_node.children) {
      SplitAST flattened_child = flatten_ast(child);
      add_flattened_child_to_parent(result, flattened_child);
    }
    return result;
  }

  SplitAST operator()(Node const &ast_node) {
    return ast_node; 
  }
};

SplitAST flatten_ast(SplitAST const &ast) {
  return mpark::visit(FlattenAST{}, ast);
}

struct ToFinalAST {
  mpark::variant<Serial, Parallel, Node> operator()(SplitASTNode const &node) {
    if (node.type == SplitType::SERIAL) {
      Serial result;
      for (mpark::variant<Serial, Parallel, Node> const &child : vector_transform(to_final_ast, node.children)) {
        if (mpark::holds_alternative<Parallel>(child)) {
          result.children.push_back(mpark::get<Parallel>(child));
        } else {
          result.children.push_back(mpark::get<Node>(child));
        }
      }
      return result;
    } else {
      Parallel result;
      for (mpark::variant<Serial, Parallel, Node> const &child : vector_transform(to_final_ast, node.children)) {
        if (mpark::holds_alternative<Serial>(child)) {
          result.children.push_back(mpark::get<Serial>(child));
        } else {
          result.children.push_back(mpark::get<Node>(child));
        }
      }
      return result;
    }
  }

  mpark::variant<Serial, Parallel, Node> operator()(Node const &node) {
    return node;
  }
};

mpark::variant<Serial, Parallel, Node> to_final_ast(SplitAST const &ast) {
  return mpark::visit(ToFinalAST{}, ast);
}

SplitASTNode::SplitASTNode(SplitType type) : type(type) {}

SplitASTNode::SplitASTNode(SplitType type, SplitAST const &child1, SplitAST const &child2) : type(type) {
  children.push_back(child1);
  children.push_back(child2);
}

SplitASTNode::SplitASTNode(SplitType type, std::vector<SplitAST> const &children) : type(type), children(children) {}

}

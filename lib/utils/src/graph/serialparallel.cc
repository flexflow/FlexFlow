#include "utils/graph/serialparallel.h"
#include "serialparallel_internal.h"
#include "utils/containers.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
namespace FlexFlow {

bool has_single_source(DiGraphView const &g) {
  return get_sources(g).size() == 1;
}

bool has_single_sink(DiGraphView const &g) {
  return get_sinks(g).size() == 1;
}

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

enum class SourceSettings { INCLUDE_SOURCE_NODES, EXCLUDE_SOURCE_NODES };

enum class SinkSettings { INCLUDE_SINK_NODES, EXCLUDE_SINK_NODES };

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

SplitAST sp_decomposition(DiGraphView const &g) {
  if (num_nodes(g) == 1) {
    return get_only(get_nodes(g));
  }

  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sinks(g);

  std::optional<Node> bottleneck = find_bottleneck_node(g);
  if (bottleneck.has_value()) {
    return SplitASTNode(SplitType::SERIAL,
                        sp_decomposition(source_to_sink_subgraph(
                            g,
                            sources,
                            {bottleneck.value()},
                            SourceSettings::INCLUDE_SOURCE_NODES,
                            SinkSettings::EXCLUDE_SINK_NODES)),
                        sp_decomposition(source_to_sink_subgraph(
                            g,
                            {bottleneck.value()},
                            sinks,
                            SourceSettings::INCLUDE_SOURCE_NODES,
                            SinkSettings::INCLUDE_SINK_NODES)));
  } else {
    return parallel_decomposition(g);
  }
}

SplitAST parallel_decomposition(DiGraphView const &g) {
  std::unordered_set<std::unordered_set<Node>> weakly_connected_components =
      get_weakly_connected_components(g);
  assert(weakly_connected_components.size() > 1);

  SplitASTNode split(SplitType::PARALLEL);
  for (auto const &component : weakly_connected_components) {
    split.children.push_back(sp_decomposition(get_subgraph(g, component)));
  }

  return split;
}

SplitASTNode::SplitASTNode(SplitType type) : SplitASTNode(type, {}) {}

SplitASTNode::SplitASTNode(SplitType type,
                           SplitAST const &lhs,
                           SplitAST const &rhs)
    : SplitASTNode(type, {lhs, rhs}) {}

SplitASTNode::SplitASTNode(SplitType type,
                           std::vector<SplitAST> const &children)
    : type(type), children(children) {}

struct FlattenAST {
  void add_flattened_child_to_parent(SplitASTNode &parent,
                                     SplitAST const &child) {
    if (std::holds_alternative<Node>(child)) {
      parent.children.push_back(child);
      return;
    }

    SplitASTNode child_node = get<SplitASTNode>(child);

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
  return visit(FlattenAST{}, ast);
}

struct ToFinalAST {
  std::variant<Serial, Parallel, Node> operator()(SplitASTNode const &node) {
    if (node.type == SplitType::SERIAL) {
      return Serial{transform(node.children, [](SplitAST const &s) {
        return narrow<std::variant<Parallel, Node>>(to_final_ast(s)).value();
      })};
    } else {
      return Parallel{transform(node.children, [](SplitAST const &s) {
        return narrow<std::variant<Serial, Node>>(to_final_ast(s)).value();
      })};
    }
  }

  std::variant<Serial, Parallel, Node> operator()(Node const &node) {
    return node;
  }
};

std::variant<Serial, Parallel, Node> to_final_ast(SplitAST const &ast) {
  return visit(ToFinalAST{}, ast);
}

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &g) {
  SplitAST ast = sp_decomposition(g);
  return to_final_ast(ast);
}

struct GetNodes {
  template <typename T>
  std::unordered_set<Node> operator()(T const &t) {
    return get_nodes(t);
  }
};

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp) {
  return visit(GetNodes{}, sp);
}

std::unordered_set<Node> get_nodes(Serial const &serial) {
  return set_union(transform(
      serial.children,
      [](std::variant<Parallel, Node> const child) -> std::unordered_set<Node> {
        return visit(GetNodes{}, child);
      }));
}

std::unordered_set<Node> get_nodes(Parallel const &parallel) {
  return set_union(
      transform(parallel.children, [](std::variant<Serial, Node> const &child) {
        return visit(GetNodes{}, child);
      }));
}

std::unordered_set<Node> get_nodes(Node const &node) {
  return {node};
}

std::unordered_map<Node, Node> parallel_extend(DiGraph &g,
                                               DiGraphView const &ext) {
  std::unordered_map<Node, Node> node_map;
  for (Node const &node : get_nodes(DiGraphView(ext))) {
    node_map.emplace(node, g.add_node());
  }
  for (auto const &edge : get_edges(ext)) {
    g.add_edge({node_map.at(edge.src), node_map.at(edge.dst)});
  }
  return node_map;
}

std::unordered_map<Node, Node> serial_extend(DiGraph &g,
                                             DiGraphView const &ext) {
  std::unordered_set<Node> original_sinks = get_sinks(g);
  std::unordered_map<Node, Node> node_map = parallel_extend(g, ext);
  for (Node const &node1 : original_sinks) {
    for (Node const &node2 : get_sources(ext)) {
      g.add_edge({node_map.at(node1), node2});
    }
  }
  return node_map;
}

std::unordered_map<Node, Node> parallel_extend(MultiDiGraph &g,
                                               MultiDiGraph const &ext) {
  std::unordered_map<Node, Node> node_map;
  std::unordered_map<NodePort, NodePort> node_port_map;
  for (Node const &node : get_nodes(MultiDiGraphView(ext))) {
    node_map.emplace(node, g.add_node());
  }
  for (NodePort const &node_port : get_present_node_ports(ext)) {
    node_port_map.emplace(node_port, g.add_node_port());
  }
  for (MultiDiEdge const &edge : get_edges(ext)) {
    g.add_edge(MultiDiEdge{node_map.at(edge.dst),
                           node_port_map.at(edge.dst_idx),
                           node_map.at(edge.src),
                           node_port_map.at(edge.src_idx)});
  }
  return node_map;
}

std::unordered_map<Node, Node> serial_extend(MultiDiGraph &g,
                                             MultiDiGraph const &ext) {
  std::unordered_set<Node> original_sinks = get_sinks(g);
  std::unordered_map<Node, Node> node_map = parallel_extend(g, ext);
  for (Node const &node1 : original_sinks) {
    for (Node const &node2 : get_sources(ext)) {
      g.add_edge(MultiDiEdge{
          node_map.at(node2), g.add_node_port(), node1, g.add_node_port()});
    }
  }
  return node_map;
}

MultiDiGraph serial_composition(MultiDiGraph const &g1,
                                MultiDiGraph const &g2) {
  MultiDiGraph g = g1;
  serial_extend(g, g2);
  return g;
}

MultiDiGraph parallel_composition(MultiDiGraph const &g1,
                                  MultiDiGraph const &g2) {
  MultiDiGraph g = g1;
  parallel_extend(g, g2);
  return g;
}

SerialParallelDecomposition parallel_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions) {
  if (sp_compositions.size() == 1) {
    return sp_compositions.at(0);
  }
  Parallel composition;
  for (SerialParallelDecomposition const &sp_comp : sp_compositions) {
    if (std::holds_alternative<Parallel>(sp_comp)) {
      for (std::variant<Serial, Node> const &children :
           std::get<Parallel>(sp_comp)
               .children) { // unwrapping the parallel node, since a Parallel
                            // cannot contain other Parallels
        composition.children.push_back(children);
      }
    } else if (std::holds_alternative<Serial>(sp_comp)) {
      composition.children.push_back(std::get<Serial>(sp_comp));
    } else {
      assert(std::holds_alternative<Node>(sp_comp));
      composition.children.push_back(std::get<Node>(sp_comp));
    }
  }
  return composition;
}

SerialParallelDecomposition serial_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions) {
  if (sp_compositions.size() == 1) {
    return sp_compositions.at(0);
  }
  Serial composition;
  for (SerialParallelDecomposition const &sp_comp : sp_compositions) {
    if (std::holds_alternative<Serial>(sp_comp)) {
      for (std::variant<Parallel, Node> const &subnode :
           std::get<Serial>(sp_comp)
               .children) { // unwrapping the serial node, since a Serial cannot
                            // contain other Serials
        composition.children.push_back(subnode);
      }
    } else if (std::holds_alternative<Parallel>(sp_comp)) {
      composition.children.push_back(std::get<Parallel>(sp_comp));
    } else {
      assert(std::holds_alternative<Node>(sp_comp));
      composition.children.push_back(std::get<Node>(sp_comp));
    }
  }
  return composition;
}

SerialParallelDecomposition
    to_sp_decomp(std::variant<Serial, Node> const &child) {
  return std::holds_alternative<Node>(child)
             ? SerialParallelDecomposition{std::get<Node>(child)}
             : SerialParallelDecomposition{std::get<Serial>(child)};
}

std::vector<SerialParallelDecomposition>
    to_sp_decomp(std::vector<std::variant<Serial, Node>> const &children) {
  return transform(
      children,
      [](std::variant<Serial, Node> const &child)
          -> SerialParallelDecomposition { return to_sp_decomp(child); });
}

SerialParallelDecomposition
    to_sp_decomp(std::variant<Parallel, Node> const &child) {
  return std::holds_alternative<Node>(child)
             ? SerialParallelDecomposition{std::get<Node>(child)}
             : SerialParallelDecomposition{std::get<Parallel>(child)};
}

std::vector<SerialParallelDecomposition>
    to_sp_decomp(std::vector<std::variant<Parallel, Node>> const &children) {
  return transform(
      children,
      [](std::variant<Parallel, Node> const &child)
          -> SerialParallelDecomposition { return to_sp_decomp(child); });
}

bool isempty(SerialParallelDecomposition const &sp) {
  if (std::holds_alternative<Node>(sp)) {
    return false;
  } else if (std::holds_alternative<Serial>(sp)) {
    return all_of(to_sp_decomp(std::get<Serial>(sp).children), isempty);
  } else {
    assert(std::holds_alternative<Parallel>(sp));
    return all_of(to_sp_decomp(std::get<Parallel>(sp).children), isempty);
  }
}

std::vector<std::variant<Parallel, Node>> filter_empty(Serial const &serial) {
  // return filter(serial.children, [](auto const &child) {return
  // !isempty(to_sp_decomp(child));});
  std::vector<std::variant<Parallel, Node>> filtered;
  for (std::variant<Parallel, Node> const &child : serial.children) {
    if (!isempty(to_sp_decomp(child))) {
      filtered.push_back(child);
    }
  }
  return filtered;
}

std::vector<std::variant<Serial, Node>> filter_empty(Parallel const &parallel) {
  // return filter(parallel.children, [](auto const &child) {return
  // !isempty(to_sp_decomp(child));});
  std::vector<std::variant<Serial, Node>> filtered;
  for (std::variant<Serial, Node> const &child : parallel.children) {
    if (!isempty(to_sp_decomp(child))) {
      filtered.push_back(child);
    }
  }
  return filtered;
}

SerialParallelDecomposition normalize(SerialParallelDecomposition const &sp) {
  if (std::holds_alternative<Node>(sp)) {
    return sp;
  }

  auto normalize_children = [](auto const &container) {
    std::vector<SerialParallelDecomposition> normalized_children;
    for (const auto &child : filter_empty(container)) {
      if (std::holds_alternative<Node>(child)) {
        normalized_children.push_back(std::get<Node>(child));
      } else {
        normalized_children.push_back(normalize(to_sp_decomp(child)));
      }
    }
    return normalized_children;
  };

  auto simplify_composition =
      [](SerialParallelDecomposition const &composition) {
        if (std::holds_alternative<Serial>(composition)) {
          Serial serial = std::get<Serial>(composition);
          if (serial.children.size() == 1) {
            return to_sp_decomp(serial.children[0]);
          }
        } else if (std::holds_alternative<Parallel>(composition)) {
          Parallel parallel = std::get<Parallel>(composition);
          if (parallel.children.size() == 1) {
            return to_sp_decomp(parallel.children[0]);
          }
        }
        return composition;
      };

  if (std::holds_alternative<Serial>(sp)) {
    std::vector<SerialParallelDecomposition> normalized_children =
        normalize_children(std::get<Serial>(sp));
    return simplify_composition(serial_composition(normalized_children));
  } else {
    assert(std::holds_alternative<Parallel>(sp));
    std::vector<SerialParallelDecomposition> normalized_children =
        normalize_children(std::get<Parallel>(sp));
    return simplify_composition(parallel_composition(normalized_children));
  }
}

std::unordered_map<Node, size_t>
    node_counter(SerialParallelDecomposition const &sp) {
  std::unordered_map<Node, size_t> counter;

  if (std::holds_alternative<Node>(sp)) {
    Node node = std::get<Node>(sp);
    counter[node]++;
  } else if (std::holds_alternative<Serial>(sp)) {
    for (std::variant<Parallel, Node> const &child :
         std::get<Serial>(sp).children) {
      std::unordered_map<Node, size_t> child_counter =
          node_counter(to_sp_decomp(child));
      for (auto const &[node, count] : child_counter) {
        counter[node] += count;
      }
    }
  } else {
    assert(std::holds_alternative<Parallel>(sp));
    for (std::variant<Serial, Node> const &child :
         std::get<Parallel>(sp).children) {
      std::unordered_map<Node, size_t> child_counter =
          node_counter(to_sp_decomp(child));
      for (auto const &[node, count] : child_counter) {
        counter[node] += count;
      }
    }
  }

  return counter;
}

size_t node_count(SerialParallelDecomposition const &sp) {
  return sum(values(node_counter(sp)));
}

size_t node_count(DiGraphView const &g) {
  return get_nodes(g).size();
}

float compute_cost(SerialParallelDecomposition const &sp,
                   std::unordered_map<Node, float> cost_map) {
  auto cost_per_node_group = [&](std::pair<Node, float> const &pair) {
    return pair.second * cost_map.at(pair.first);
  };
  std::unordered_map<Node, size_t> counter = node_counter(sp);
  std::vector<std::pair<Node, size_t>> pairs(counter.cbegin(), counter.cend());
  return sum(transform(pairs, cost_per_node_group));
}

float compute_cost(DiGraphView const &g,
                   std::unordered_map<Node, float> const &cost_map) {
  return sum(transform(as_vector(get_nodes(g)),
                       [&](Node const &node) { return cost_map.at(node); }));
}

float critical_path_cost(DiGraphView const &g,
                         std::unordered_map<Node, float> const &cost_map) {
  return maximum(
      values(get_weighted_longest_path_lengths_from_source_node(g, cost_map)));
}

float critical_path_cost(SerialParallelDecomposition const &sp,
                         std::unordered_map<Node, float> const &cost_map) {
  if (std::holds_alternative<Node>(sp)) {
    return cost_map.at(std::get<Node>(sp));
  } else if (std::holds_alternative<Serial>(sp)) {
    return sum(transform(std::get<Serial>(sp).children,
                         [&](std::variant<Parallel, Node> const &child) {
                           return critical_path_cost(to_sp_decomp(child),
                                                     cost_map);
                         }));
  } else {
    assert(std::holds_alternative<Parallel>(sp));
    return maximum(transform(std::get<Parallel>(sp).children,
                             [&](std::variant<Serial, Node> const &child) {
                               return critical_path_cost(to_sp_decomp(child),
                                                         cost_map);
                             }));
  }
}

float average_parallelism_degree(
    SerialParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map) {
  NOT_IMPLEMENTED();
}

float max_parallelism_degree(SerialParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map) {
  NOT_IMPLEMENTED();
}

// Metrics

float relative_cost_increase(DiGraphView const &g,
                             SerialParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map) {
  return compute_cost(sp, cost_map) / compute_cost(g, cost_map);
}

float relative_critical_path_cost_increase(
    DiGraphView const &g,
    SerialParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map) {
  return critical_path_cost(sp, cost_map) / critical_path_cost(g, cost_map);
}

struct MultiDiGraphFromSPDecompositionFunctor {
  template <typename T>
  MultiDiGraph operator()(T const &t) {
    return multidigraph_from_sp_decomposition(t);
  }
};

MultiDiGraph multidigraph_from_sp_decomposition(
    SerialParallelDecomposition const &sp_decomposition) {
  return visit(MultiDiGraphFromSPDecompositionFunctor{}, sp_decomposition);
}

MultiDiGraph multidigraph_from_sp_decomposition(
    std::variant<Parallel, Node> const &sp_decomposition) {
  return visit(MultiDiGraphFromSPDecompositionFunctor{}, sp_decomposition);
}

MultiDiGraph multidigraph_from_sp_decomposition(
    std::variant<Serial, Node> const &sp_decomposition) {
  return visit(MultiDiGraphFromSPDecompositionFunctor{}, sp_decomposition);
}

MultiDiGraph multidigraph_from_sp_decomposition(Serial const &serial) {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
  for (std::variant<Parallel, Node> const &child : serial.children) {
    serial_extend(g, multidigraph_from_sp_decomposition(child));
  }
  return g;
}

MultiDiGraph multidigraph_from_sp_decomposition(Parallel const &parallel) {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
  for (std::variant<Serial, Node> const &child : parallel.children) {
    parallel_extend(g, multidigraph_from_sp_decomposition(child));
  }
  return g;
}

MultiDiGraph multidigraph_from_sp_decomposition(Node const &Node) {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
  g.add_node();
  return g;
}

bool Parallel::operator==(Parallel const &other) const {
  return as_set(children) == as_set(other.children);
}

bool Parallel::operator!=(Parallel const &other) const {
  return !(*this == other);
}

} // namespace FlexFlow

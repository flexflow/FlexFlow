#include "utils/graph/algorithms.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/get_only.h"
#include "utils/containers/intersection.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_node_with_greatest_topo_rank.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/traversal.h"
#include "utils/graph/undirected/undirected_edge_query.h"
#include "utils/graph/views/views.h"
#include "utils/hash-utils.h"
#include "utils/variant.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

namespace FlexFlow {

template <typename G>
static std::vector<Node> add_nodes_impl(G &g, int num_nodes) {
  std::vector<Node> nodes;
  for (int i = 0; i < num_nodes; i++) {
    nodes.push_back(g.add_node());
  }
  return nodes;
}

std::vector<Node> add_nodes(Graph &g, int num_nodes) {
  return add_nodes_impl<Graph>(g, num_nodes);
}

std::vector<Node> add_nodes(UndirectedGraph &g, int num_nodes) {
  return add_nodes_impl<UndirectedGraph>(g, num_nodes);
}

std::vector<Node> add_nodes(DiGraph &g, int num_nodes) {
  return add_nodes_impl<DiGraph>(g, num_nodes);
}

// std::vector<Node> add_nodes(MultiDiGraph &g, int num_nodes) {
//   return add_nodes_impl<MultiDiGraph>(g, num_nodes);
// }

// std::vector<Node> add_nodes(OpenMultiDiGraph &g, int num_nodes) {
//   return add_nodes_impl<OpenMultiDiGraph>(g, num_nodes);
// }

// std::vector<NodePort> add_node_ports(MultiDiGraph &g, int num_node_ports) {
//   std::vector<NodePort> node_ports;
//   for (int i = 0; i < num_node_ports; i++) {
//     node_ports.push_back(g.add_node_port());
//   }
//   return node_ports;
// }

// std::unordered_set<Node> get_nodes(InputMultiDiEdge const &edge) {
//   return {edge.dst};
// }

// std::unordered_set<Node> get_nodes(OutputMultiDiEdge const &edge) {
//   return {edge.src};
// }

// std::unordered_set<Node> get_nodes(MultiDiEdge const &edge) {
//   return {edge.src, edge.dst};
// }

struct GetNodesFunctor {
  template <typename T>
  std::unordered_set<Node> operator()(T const &t) {
    return get_nodes(t);
  }
};

// std::unordered_set<Node> get_nodes(OpenMultiDiEdge const &edge) {
//   return visit(GetNodesFunctor{}, edge);
// }

std::unordered_set<Node> query_nodes(GraphView const &g,
                                     std::unordered_set<Node> const &nodes) {
  return g.query_nodes(NodeQuery{nodes});
}

// std::unordered_set<NodePort> get_present_node_ports(MultiDiGraphView const
// &g) {
//   return flatmap(get_edges(g), [](MultiDiEdge const &e) {
//     return std::unordered_set<NodePort>{e.src_idx, e.dst_idx};
//   });
// }

// void remove_node(MultiDiGraph &g, Node const &n) {
//   for (MultiDiEdge const &e : get_incoming_edges(g, n)) {
//     g.remove_edge(e);
//   }
//   for (MultiDiEdge const &e : get_outgoing_edges(g, n)) {
//     g.remove_edge(e);
//   }
//   g.remove_node_unsafe(n);
// }

void remove_node(DiGraph &g, Node const &n) {
  for (DirectedEdge const &e : get_incoming_edges(g, n)) {
    g.remove_edge(e);
  }
  for (DirectedEdge const &e : get_outgoing_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
}

void remove_node(UndirectedGraph &g, Node const &n) {
  for (UndirectedEdge const &e : get_node_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
}

// void remove_node_if_unused(MultiDiGraph &g, Node const &n) {
//   if (!get_incoming_edges(g, n).empty()) {
//     return;
//   }
//   if (!get_outgoing_edges(g, n).empty()) {
//     return;
//   }
//
//   g.remove_node_unsafe(n);
// }

void remove_node_if_unused(DiGraph &g, Node const &n) {
  if (!get_incoming_edges(g, n).empty()) {
    return;
  }
  if (!get_outgoing_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

void remove_node_if_unused(UndirectedGraph &g, Node const &n) {
  if (!get_node_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

// void add_edges(MultiDiGraph &g, std::vector<MultiDiEdge> const &edges) {
//   for (MultiDiEdge const &e : edges) {
//     g.add_edge(e);
//   }
// }

void add_edges(DiGraph &g, std::vector<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

void add_edges(UndirectedGraph &g, std::vector<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

// bool contains_edge(MultiDiGraphView const &g, MultiDiEdge const &e) {
//   return contains(g.query_edges({e.src, e.dst, e.src_idx, e.dst_idx}), e);
// }

bool contains_edge(DiGraphView const &g, DirectedEdge const &e) {
  return contains(g.query_edges(DirectedEdgeQuery{e.src, e.dst}), e);
}

bool contains_edge(UndirectedGraphView const &g, UndirectedEdge const &e) {
  UndirectedEdgeQuery q =
      UndirectedEdgeQuery{{e.endpoints.max(), e.endpoints.min()}};
  return contains(g.query_edges(q), e);
}

// void remove_edges(MultiDiGraph &g,
//                   std::unordered_set<MultiDiEdge> const &edges) {
//   for (MultiDiEdge const &e : edges) {
//     assert(contains_edge(g, e));
//     g.remove_edge(e);
//   }
// }

void remove_edges(DiGraph &g, std::unordered_set<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    assert(contains_edge(g, e));
    g.remove_edge(e);
  }
}

void remove_edges(UndirectedGraph &g,
                  std::unordered_set<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    assert(contains_edge(g, e));
    g.remove_edge(e);
  }
}

std::unordered_set<Node> get_endpoints(UndirectedEdge const &e) {
  return {e.endpoints.min(), e.endpoints.max()};
}

// std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &g) {
//   return g.query_edges(MultiDiEdgeQuery::all());
// }

std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &g) {
  return g.query_edges(undirected_edge_query_all());
}

// std::unordered_set<OpenMultiDiEdge> get_edges(OpenMultiDiGraphView const &g)
// {
//   return g.query_edges(OpenMultiDiEdgeQuery::all());
// }

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                                  Node const &n) {
  return g.query_edges(UndirectedEdgeQuery{n});
}

// std::unordered_set<MultiDiOutput> get_outputs(MultiDiGraphView const &g) {
//   return transform(get_edges(g), [&](MultiDiEdge const &e) -> MultiDiOutput {
//     return static_cast<MultiDiOutput>(e);
//   });
// }

// std::unordered_set<MultiDiInput> get_inputs(MultiDiGraphView const &g) {
//   return transform(get_edges(g), [&](MultiDiEdge const &e) -> MultiDiInput {
//     return static_cast<MultiDiInput>(e);
//   });
// }

// std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &g,
//                                                    Node const &n) {
//   return get_incoming_edges(g, std::unordered_set<Node>{n});
// }

// std::unordered_set<MultiDiEdge>
//     get_incoming_edges(MultiDiGraphView const &g,
//                        std::unordered_set<Node> dsts) {
//   return g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes(dsts));
// }

// std::unordered_set<MultiDiEdge>
//     get_outgoing_edges(MultiDiGraphView const &g,
//                        std::unordered_set<Node> const &srcs) {
//   return g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(srcs));
// }

// std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g,
//                                                    Node const &n) {
//   return get_outgoing_edges(g, std::unordered_set<Node>{n});
// }

// std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
//     get_incoming_edges_by_idx(MultiDiGraphView const &g, Node const &n) {
//   std::unordered_set<MultiDiEdge> edges = get_incoming_edges(g, n);
//   std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>> result;
//   for (MultiDiEdge const &e : edges) {
//     result[e.dst_idx].insert(e);
//   }
//   return result;
// }

// std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
//     get_outgoing_edges_by_idx(MultiDiGraphView const &g, Node const &n) {
//   std::unordered_set<MultiDiEdge> edges = get_outgoing_edges(g, n);
//   std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>> result;
//   for (MultiDiEdge const &e : edges) {
//     result[e.src_idx].insert(e);
//   }
//   return result;
// }

// std::unordered_set<DownwardOpenMultiDiEdge>
//     get_outgoing_edges(OpenMultiDiGraphView const &g, Node const &n) {
//   return value_all(
//       narrow<DownwardOpenMultiDiEdge>(g.query_edges(OpenMultiDiEdgeQuery(
//           InputMultiDiEdgeQuery::none(),
//           MultiDiEdgeQuery::all().with_src_nodes({n}),
//           OutputMultiDiEdgeQuery::all().with_src_nodes({n})))));
// }

// std::unordered_set<UpwardOpenMultiDiEdge>
//     get_incoming_edges(OpenMultiDiGraphView const &g, Node const &n) {
//   return value_all(narrow<UpwardOpenMultiDiEdge>(g.query_edges(
//       OpenMultiDiEdgeQuery(InputMultiDiEdgeQuery::all().with_dst_nodes({n}),
//                            MultiDiEdgeQuery::all().with_dst_nodes({n}),
//                            OutputMultiDiEdgeQuery::none()))));
// }

// std::unordered_set<OutputMultiDiEdge>
//     get_open_outputs(OpenMultiDiGraphView const &g) {
//   return narrow<OutputMultiDiEdge>(
//       g.query_edges(OutputMultiDiEdgeQuery::all()));
// }

// std::unordered_set<InputMultiDiEdge>
//     get_open_inputs(OpenMultiDiGraphView const &g) {
//   return
//   narrow<InputMultiDiEdge>(g.query_edges(InputMultiDiEdgeQuery::all()));
// }

std::vector<Node> get_unchecked_dfs_ordering(
    DiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  UncheckedDFSView dfs_view = unchecked_dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node>
    get_dfs_ordering(DiGraphView const &g,
                     std::unordered_set<Node> const &starting_points) {
  CheckedDFSView dfs_view = dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node>
    get_bfs_ordering(DiGraphView const &g,
                     std::unordered_set<Node> const &starting_points) {
  BFSView bfs_view = bfs(g, starting_points);
  return {bfs_view.begin(), bfs_view.end()};
}

// std::optional<bool> is_acyclic(MultiDiGraph const &g) {
//   return is_acyclic(g);
// }

std::vector<DirectedEdge> get_edge_topological_ordering(DiGraphView const &g) {
  std::vector<DirectedEdge> result;
  for (Node const &n : get_topological_ordering(g)) {
    for (DirectedEdge const &e : get_outgoing_edges(g, n)) {
      result.push_back(e);
    }
  }

  assert(result.size() == get_edges(g).size());

  return result;
}

// Node get_src_node(MultiDiEdge const &e) {
//   return e.src;
// }

// Node get_dst_node(MultiDiEdge const &e) {
//   return e.dst;
// }

// Node get_dst_node(InputMultiDiEdge const &e) {
//   return e.dst;
// }

// Node get_src_node(OutputMultiDiEdge const &e) {
//   return e.src;
// }

// NodePort get_src_idx(MultiDiEdge const &e) {
//   return e.src_idx;
// }

// NodePort get_dst_idx(MultiDiEdge const &e) {
//   return e.dst_idx;
// }

// NodePort get_dst_idx(InputMultiDiEdge const &e) {
//   return e.dst_idx;
// }

// NodePort get_src_idx(OutputMultiDiEdge const &e) {
//   return e.src_idx;
// }

std::unordered_set<Node> get_neighbors(DiGraphView const &g, Node const &n) {
  UndirectedGraphView undirected = as_undirected(g);
  return get_neighbors(undirected, n);
}

// std::unordered_set<Node> get_neighbors(MultiDiGraphView const &g,
//                                        Node const &n) {
//   UndirectedGraphView undirected = as_undirected(g);
//   return get_neighbors(undirected, n);
// }

std::unordered_set<Node> get_neighbors(UndirectedGraphView const &g,
                                       Node const &n) {
  return flatmap(get_node_edges(g, n), [&](UndirectedEdge const &edge) {
    return set_difference(get_endpoints(edge), {n});
  });
}

// std::vector<MultiDiEdge>
//     get_edge_topological_ordering(MultiDiGraphView const &g) {
//   std::vector<MultiDiEdge> result;
//   for (Node const &n : get_topological_ordering(g)) {
//     for (MultiDiEdge const &e : get_outgoing_edges(g, n)) {
//       result.push_back(e);
//     }
//   }
//
//   assert(result.size() == get_edges(g).size());
//
//   return result;
// }

// std::pair<OutputMultiDiEdge, InputMultiDiEdge>
//     split_edge(MultiDiEdge const &e) {
//   return {OutputMultiDiEdge{e.src, e.src_idx, e.get_uid()},
//           InputMultiDiEdge{e.dst, e.dst_idx, e.get_uid()}};
// }

// MultiDiEdge unsplit_edge(OutputMultiDiEdge const &output_edge,
//                          InputMultiDiEdge const &input_edge) {
//   assert(output_edge.uid.first == input_edge.dst.value());
//   assert(output_edge.uid.second == input_edge.dst_idx.value());
//   assert(input_edge.uid.first == output_edge.src.value());
//   assert(input_edge.uid.second == output_edge.src_idx.value());
//   return {
//       input_edge.dst, input_edge.dst_idx, output_edge.src,
//       output_edge.src_idx};
// }

// std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &g,
//                                             GraphSplit const &s) {
//   return set_union(
//       g.query_edges(
//           MultiDiEdgeQuery::all().with_src_nodes(s.first).with_dst_nodes(
//               s.second)),
//       g.query_edges(
//           MultiDiEdgeQuery::all().with_src_nodes(s.second).with_dst_nodes(
//               s.first)));
// }

// std::unordered_set<MultiDiEdge>
//     get_cut_set(MultiDiGraphView const &g,
//                 std::unordered_set<Node> const &nodes) {
//   return get_cut_set(g, GraphSplit{nodes, set_difference(get_nodes(g),
//   nodes)});
// }

// bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>>
//     get_edge_splits(MultiDiGraphView const &graph, GraphSplit const &split) {
//   bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>> result;
//   std::unordered_set<MultiDiEdge> cut_set = get_cut_set(graph, split);
//   for (MultiDiEdge const &edge : cut_set) {
//     result.equate(edge, split_edge(edge));
//   }
//   return result;
//   return generate_bidict(get_cut_set(graph, split),
//                          [](MultiDiEdge const &e) { return split_edge(e); });
// }

UndirectedGraphView get_subgraph(UndirectedGraphView const &g,
                                 std::unordered_set<Node> const &nodes) {
  return UndirectedGraphView::create<UndirectedSubgraphView>(g, nodes);
}

DiGraphView get_subgraph(DiGraphView const &g,
                         std::unordered_set<Node> const &nodes) {
  return DiGraphView::create<DiSubgraphView>(g, nodes);
}

// MultiDiGraphView get_subgraph(MultiDiGraphView const &g,
//                               std::unordered_set<Node> const &nodes) {
//   return MultiDiGraphView::create<MultiDiSubgraphView>(g, nodes);
// }

// MultiDiGraphView join(MultiDiGraphView const &lhs,
//                       MultiDiGraphView const &rhs) {
//   return MultiDiGraphView::create<JoinedMultiDigraphView>(lhs, rhs);
// }

DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs) {
  return DiGraphView::create<JoinedDigraphView>(lhs, rhs);
}

UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs) {
  return UndirectedGraphView::create<JoinedUndirectedGraphView>(lhs, rhs);
}

UndirectedGraphView as_undirected(DiGraphView const &g) {
  return UndirectedGraphView::create<ViewDiGraphAsUndirectedGraph>(g);
}

// MultiDiGraphView as_multidigraph(DiGraphView const &g) {
//   return MultiDiGraphView::create<ViewDiGraphAsMultiDiGraph>(g);
// }

DiGraphView as_digraph(UndirectedGraphView const &g) {
  return DiGraphView::create<ViewUndirectedGraphAsDiGraph>(g);
}

// OpenMultiDiGraphView as_openmultidigraph(MultiDiGraphView const &g) {
//   return OpenMultiDiGraphView::create<ViewMultiDiGraphAsOpenMultiDiGraph>(g);
// }

// std::unordered_set<Node> get_closed_sources(OpenMultiDiGraphView const &g) {
//   return filter(get_nodes(g), [&](Node const &n) {
//     return get_incoming_edges(g, n).size() == 0;
//   });
// }

// std::unordered_set<Node> get_closed_sinks(OpenMultiDiGraphView const &g) {
//   return filter(get_nodes(g), [&](Node const &n) {
//     return get_outgoing_edges(g, n).size() == 0;
//   });
// }

// std::unordered_set<Node> get_open_sources(OpenMultiDiGraphView const &g) {
//   return filter(get_nodes(g), [&](Node const &n) {
//     return !get_incoming_edges(g, n).empty();
//   });
// }

// std::unordered_set<Node> get_open_sinks(OpenMultiDiGraphView const &g) {
//   return filter(get_nodes(g), [&](Node const &n) {
//     return !get_outgoing_edges(g, n).empty();
//   });
// }

} // namespace FlexFlow

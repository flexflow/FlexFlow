#include "utils/graph/digraph/algorithms.h"
#include "utils/containers.h"
#include "utils/containers/group_by.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/traversal.h"
#include "utils/graph/views/views.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g) {
  return g.query_edges(directed_edge_query_all());
}

std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>::matchall(),
      query_set<Node>{n},
  });
}

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_incoming_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
      group_by(g.query_edges(DirectedEdgeQuery{
                   query_set<Node>::matchall(),
                   query_set<Node>{ns},
               }),
               [](DirectedEdge const &e) { return e.dst; });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>{n},
      query_set<Node>::matchall(),
  });
}

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_outgoing_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
      group_by(g.query_edges(DirectedEdgeQuery{
                   query_set<Node>::matchall(),
                   query_set<Node>{ns},
               }),
               [](DirectedEdge const &e) { return e.src; });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

std::unordered_set<Node> get_sources(DiGraphView const &g) {
  std::unordered_set<Node> all_nodes = get_nodes(g);
  std::unordered_set<Node> with_incoming_edge =
      transform(get_edges(g), [](DirectedEdge const &e) { return e.dst; });

  return set_minus(all_nodes, with_incoming_edge);
}

std::unordered_set<Node> get_sinks(DiGraphView const &g) {
  return get_sources(flipped(g));
}

static std::vector<Node>
    get_unchecked_topological_ordering(DiGraphView const &g) {
  auto dfs_view = unchecked_dfs(g, get_sources(g));
  std::vector<Node> order;
  std::unordered_set<Node> seen;
  std::unordered_map<Node, std::unordered_set<Node>> predecessors =
      get_predecessors(g, get_nodes(g));

  auto all_predecessors_seen = [&](Node const &n) -> bool {
    bool result = true;
    for (Node const &pred : predecessors.at(n)) {
      result &= contains(seen, pred);
    }
    return result;
  };

  unchecked_dfs_iterator it = dfs_view.cbegin();
  while (it != dfs_view.cend()) {
    if (all_predecessors_seen(*it)) {
      order.push_back(*it);
      seen.insert(*it);
      it++;
    } else {
      it.skip();
    }
  }

  return order;
}

std::vector<Node> get_topological_ordering(DiGraphView const &g) {
  assert(is_acyclic(g));
  return get_unchecked_topological_ordering(g);
}

std::optional<bool> is_acyclic(DiGraphView const &g) {
  if (num_nodes(g) == 0) {
    return std::nullopt;
  }
  std::unordered_set<Node> sources = get_sources(g);
  if (sources.size() == 0) {
    return false;
  }
  auto dfs_view = unchecked_dfs(g, sources);
  std::unordered_set<Node> seen;
  for (unchecked_dfs_iterator it = dfs_view.begin(); it != dfs_view.end();
       it++) {
    if (contains(seen, *it)) {
      return false;
    } else {
      seen.insert(*it);
    }
  }
  if (seen != get_nodes(g)) {
    return false;
  }
  return true;
}

DiGraphView flipped(DiGraphView const &g) {
  return DiGraphView::create<FlippedView>(g);
}

std::unordered_set<Node> get_predecessors(DiGraphView const &g, Node const &n) {
  return get_predecessors(g, std::unordered_set<Node>{n}).at(n);
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &g, std::unordered_set<Node> const &ns) {
  return map_values(get_incoming_edges(g, ns),
                    [](std::unordered_set<DirectedEdge> const &es) {
                      return transform(
                          es, [](DirectedEdge const &e) { return e.src; });
                    });
}

} // namespace FlexFlow

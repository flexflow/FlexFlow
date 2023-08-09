#ifndef _DOMINATORS_H
#define _DOMINATORS_H

#include "basic_graph.h"
#include "graph_structures.h"
#include "tl/optional.hpp"
#include "utils/dot_file.h"
#include "utils/record_formatter.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <queue>

namespace FlexFlow {
namespace PCG {
namespace Utils {

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_set<typename Structure::vertex_type> nodes(G const &g) {
  Structure s;

  return s.get_nodes(g);
}

template <typename G, typename Structure = GraphStructure<G>>
bool has_edge(G const &g,
              typename Structure::vertex_type const &src,
              typename Structure::vertex_type const &dst) {
  Structure s;

  for (auto const &e : s.get_outgoing_edges(g, src)) {
    if (s.get_dst(g, e) == dst) {
      return true;
    }
  }

  return false;
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_set<typename Structure::edge_type>
    outgoing_edges(G const &g, typename Structure::vertex_type const &n) {
  Structure s;
  return s.get_outgoing_edges(g, n);
}

template <typename G, typename Structure = GraphStructure<G>>
std::pair<typename Structure::vertex_type, typename Structure::vertex_type>
    get_basic_edge(G const &g, typename Structure::edge_type const &e) {
  Structure s;

  return {s.get_src(g, e), s.get_dst(g, e)};
}

template <typename G, typename Structure = GraphStructure<G>>
std::vector<typename Structure::edge_type> get_edges(G const &g) {
  using N = typename Structure::vertex_type;
  using E = typename Structure::edge_type;

  Structure s;

  std::vector<E> edges;

  for (N const &n : s.get_nodes(g)) {
    for (E const &e : s.get_outgoing_edges(g, n)) {
      edges.push_back(e);
    }
  }

  return edges;
}

template <typename G, typename Structure = GraphStructure<G>>
void successors(G const &g,
                typename Structure::vertex_type const &node,
                std::unordered_set<typename Structure::vertex_type> *succ) {
  Structure s;
  for (auto const &edge : s.get_outgoing_edges(g, node)) {
    succ->insert(s.get_dst(g, edge));
  }
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_set<typename Structure::vertex_type>
    successors(G const &g, typename Structure::vertex_type const &node) {
  // using N = typename Structure::vertex_type;

  std::unordered_set<typename Structure::vertex_type> succ;
  successors<G, Structure>(g, node, &succ);

  return succ;
}

template <typename G, typename Structure = GraphStructure<G>>
tl::optional<typename Structure::vertex_type>
    successor(G const &g, typename Structure::vertex_type const &node) {
  auto succs = successors<G, Structure>(g, node);
  if (succs.size() == 1) {
    return *succs.begin();
  } else {
    return tl::nullopt;
  }
}

template <typename G, typename Structure = GraphStructure<G>>
void predecessors(G const &g,
                  typename Structure::vertex_type const &node,
                  std::unordered_set<typename Structure::vertex_type> *pred) {
  Structure s;
  for (auto const &edge : s.get_incoming_edges(g, node)) {
    pred->insert(s.get_src(g, edge));
  }
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_set<typename Structure::vertex_type>
    predecessors(G const &g, typename Structure::vertex_type const &node) {
  // using N = typename Structure::vertex_type;

  std::unordered_set<typename Structure::vertex_type> pred;
  predecessors<G, Structure>(g, node, &pred);

  return pred;
}

template <typename G, typename Structure = GraphStructure<G>>
tl::optional<typename Structure::vertex_type>
    predecessor(G const &g, typename Structure::vertex_type const &node) {
  auto preds = predecessors<G, Structure>(g, node);
  if (preds.size() == 1) {
    return *preds.begin();
  } else {
    return tl::nullopt;
  }
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_set<typename Structure::vertex_type> roots(G const &g) {
  using N = typename Structure::vertex_type;

  Structure s;

  std::unordered_set<N> nodes = s.get_nodes(g);
  std::unordered_set<N> roots;
  for (auto const &node : nodes) {
    if (s.get_incoming_edges(g, node).empty()) {
      roots.insert(node);
    }
  }

  return roots;
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_set<typename Structure::vertex_type> leaves(G const &g) {
  return roots<G, ReverseStructure<Structure>>(g);
}

template <typename G, typename Structure = GraphStructure<G>>
void topo_sort(G const &g,
               std::vector<typename Structure::vertex_type> *ordering) {
  using N = typename Structure::vertex_type;

  Structure s;
  std::unordered_map<N, std::unordered_set<N>> predecessors;

  std::queue<N> q;
  for (auto const &node : s.get_nodes(g)) {
    predecessors[node];
    for (auto const &edge : s.get_incoming_edges(g, node)) {
      predecessors.at(node).insert(s.get_src(g, edge));
    }
  }

  for (auto it = predecessors.begin(); it != predecessors.end();) {
    if (it->second.empty()) {
      q.push(it->first);
      it = predecessors.erase(it);
    } else {
      it++;
    }
  }

  std::unordered_set<N> node_successors;
  while (!q.empty()) {
    N const &current = q.front();

    ordering->push_back(current);

    node_successors.clear();
    successors<G, Structure>(g, current, &node_successors);
    for (auto const &succ : node_successors) {
      if (predecessors.find(succ) != predecessors.end()) {
        predecessors.at(succ).erase(current);
        if (predecessors.at(succ).empty()) {
          predecessors.erase(succ);
          q.push(succ);
        }
      }
    }

    q.pop();
  }
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_map<typename Structure::vertex_type,
                   std::unordered_set<typename Structure::vertex_type>>
    dominators(G const &g) {
  using N = typename Structure::vertex_type;
  // using E = typename Structure::edge_type;

  // Structure s;

  std::vector<N> nodes;
  topo_sort<G, Structure>(g, &nodes);
  std::unordered_map<N, std::unordered_set<N>> dom;

  std::unordered_set<N> pred_part;
  for (auto const &node : nodes) {
    pred_part.clear();
    predecessors<G, Structure>(g, node, &pred_part);
    for (auto const &p : pred_part) {
      if (dom.find(node) == dom.end()) {
        dom[node] = dom.at(p);
      } else {
        auto &node_dom_set = dom.at(node);
        auto const &p_dom_set = dom.at(p);
        for (auto it = node_dom_set.begin(); it != node_dom_set.end();) {
          if (p_dom_set.find(*it) == p_dom_set.end()) {
            it = node_dom_set.erase(it);
          } else {
            it++;
          }
        }
      }
    }
    dom[node].insert(node);
  }

  return dom;
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_map<typename Structure::vertex_type,
                   std::unordered_set<typename Structure::vertex_type>>
    post_dominators(G const &g) {
  return dominators<G, ReverseStructure<Structure>>(g);
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_map<typename Structure::vertex_type,
                   typename Structure::vertex_type>
    imm_dominators(G const &g) {
  using N = typename Structure::vertex_type;
  // using E = typename Structure::edge_type;

  std::vector<N> topo;
  topo_sort<G, Structure>(g, &topo);
  std::unordered_map<N, int> topo_rank;
  for (int i = 0; i < (int)topo.size(); i++) {
    topo_rank[topo[i]] = i;
  }
  std::unordered_map<N, std::unordered_set<N>> dom =
      dominators<G, Structure>(g);

  std::unordered_map<N, N> imm_dom;
  for (auto const &kv : dom) {
    N const &n = kv.first;
    std::unordered_set<N> const &n_doms = kv.second;

    // if a node is only dominated by itself, set the dominator to itself to
    // signify that it has no immediate dominator
    if (n_doms.size() == 1) {
      imm_dom[n] = n;
      continue;
    }

    N const *n_imm_dom = nullptr;
    int current_topo_rank = std::numeric_limits<int>::min();
    for (auto const &d : n_doms) {
      if (topo_rank.at(d) > current_topo_rank && d != n) {
        n_imm_dom = &d;
        current_topo_rank = topo_rank.at(d);
      }
    }
    imm_dom[n] = *n_imm_dom;
  }

  return imm_dom;
}

template <typename G, typename Structure = GraphStructure<G>>
void dfs(G const &g,
         typename Structure::vertex_type const &n,
         std::function<void(G const &,
                            Structure const &,
                            typename Structure::vertex_type const &,
                            typename Structure::vertex_type const &)> const
             &visitor) {
  using N = typename Structure::vertex_type;
  using E = typename Structure::edge_type;

  Structure s;

  /* auto i_visitor = std::bind(visitor, g, s, n); */
  auto i_visitor = [&](N const &nn) { return visitor(g, s, n, nn); };

  std::queue<N> q;
  std::unordered_set<N> visited;

  auto visit = [&](N const &n) {
    if (visited.find(n) == visited.end()) {
      q.push(n);
      visited.insert(n);
    }
  };

  visit(n);

  while (!q.empty()) {
    N current = q.front();
    q.pop();

    i_visitor(current);

    for (E const &edge : s.get_outgoing_edges(g, current)) {
      N const &dst = s.get_dst(g, edge);
      visit(dst);
    }
  }

  return;
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_set<typename Structure::vertex_type>
    descendants(G const &g, typename Structure::vertex_type const &n) {
  using N = typename Structure::vertex_type;
  using E = typename Structure::edge_type;

  std::unordered_set<N> descendants;

  auto dfs_visitor = [&](G const &gg,
                         Structure const &ss,
                         N const &dfs_src,
                         N const &current_node) {
    descendants.insert(current_node);
  };

  dfs<G, Structure>(g, n, dfs_visitor);

  return descendants;
}

template <typename G, typename Structure = GraphStructure<G>>
std::vector<std::unordered_set<typename Structure::vertex_type>>
    weakly_connected_components(G const &g) {
  using N = typename Structure::vertex_type;
  using E = typename Structure::edge_type;

  std::vector<std::unordered_set<N>> result;
  std::unordered_set<N> seen;

  for (N const &n : nodes<G, UndirectedStructure<G, Structure>>(g)) {
    if (seen.find(n) != seen.end()) {
      continue;
    }

    auto component = descendants<G, UndirectedStructure<G, Structure>>(g, n);
    seen.insert(component.begin(), component.end());
    result.emplace_back(component);
  }

  return result;
}

template <typename G, typename Structure = GraphStructure<G>>
std::unordered_map<typename Structure::vertex_type,
                   typename Structure::vertex_type>
    imm_post_dominators(G const &g) {
  return imm_dominators<G, ReverseStructure<Structure>>(g);
}

template <typename G, typename Structure = GraphStructure<G>>
BasicGraph<typename Structure::vertex_type> transitive_reduction(G const &g) {
  using N = typename Structure::vertex_type;
  using E = typename Structure::edge_type;

  Structure s;
  BasicGraph<N> reduction;

  std::unordered_set<N> nodes = s.get_nodes(g);

  reduction.add_nodes(nodes);

  std::unordered_set<std::pair<N, N>> to_delete;

  auto dfs_visitor = [&](N const &src,
                         G const &gg,
                         Structure const &ss,
                         N const &dfs_src,
                         N const &nn) {
    if (nn != dfs_src && to_delete.find({src, nn}) == to_delete.end() &&
        has_edge<G, Structure>(gg, src, nn)) {
      to_delete.insert({src, nn});
    }
  };

  for (N const &n : nodes) {
    /* auto n_dfs_visitor = std::bind(dfs_visitor, n); */
    auto n_dfs_visitor =
        [&](G const &gg, Structure const &ss, N const &dfs_src, N const &nn) {
          return dfs_visitor(n, gg, ss, dfs_src, nn);
        };

    for (N const &child : successors<G, Structure>(g, n)) {
      dfs<G, Structure>(g, child, n_dfs_visitor);
    }
  }

  for (E const &e : get_edges<G, Structure>(g)) {
    std::pair<N, N> basic_edge = get_basic_edge<G, Structure>(g, e);

    if (to_delete.find(basic_edge) == to_delete.end()) {
      reduction.add_edge(basic_edge);
    }
  }

  return reduction;
}

template <typename N>
void inplace_transitive_reduction(BasicGraph<N> &g) {
  using Structure = GraphStructure<BasicGraph<N>>;
  using G = BasicGraph<N>;
  using E = std::pair<N, N>;

  std::unordered_set<E> to_delete;

  auto dfs_visitor = [&](N const &src,
                         G const &gg,
                         Structure const &ss,
                         N const &dfs_src,
                         N const &nn) {
    if (nn != dfs_src && to_delete.find({src, nn}) == to_delete.end() &&
        has_edge(gg, src, nn)) {
      to_delete.insert({src, nn});
    }
  };

  for (N const &n : g.nodes) {
    auto n_dfs_visitor =
        [&](G const &gg, Structure const &ss, N const &dfs_src, N const &nn) {
          return dfs_visitor(n, gg, ss, dfs_src, nn);
        };

    for (N const &child : successors(g, n)) {
      dfs<G, Structure>(g, child, n_dfs_visitor);
    }
  }

  for (E const &e : to_delete) {
    g.remove_edge(e);
  }
};

template <typename G, typename Structure = GraphStructure<G>>
void export_as_dot(
    DotFile<typename Structure::vertex_type> &dotfile,
    G const &g,
    std::function<RecordFormatter(typename Structure::vertex_type)> const
        &pretty) {
  using N = typename Structure::vertex_type;
  using E = typename Structure::edge_type;

  GraphStructure<G> s;

  for (N const &n : s.get_nodes(g)) {
    dotfile.add_record_node(n, pretty(n));

    for (E const &edge : s.get_incoming_edges(g, n)) {
      dotfile.add_edge(s.get_src(g, edge), s.get_dst(g, edge));
    }
  }

  dotfile.close();
}

} // namespace Utils
} // namespace PCG
} // namespace FlexFlow

#endif // _DOMINATORS_H

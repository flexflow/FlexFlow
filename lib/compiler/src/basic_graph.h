#ifndef _BASIC_GRAPH_H
#define _BASIC_GRAPH_H

#include "utils/hash-utils.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {
namespace PCG {
namespace Utils {

template <typename G>
struct GraphStructure;
/*
{
  using graph_type = ...;
  using node_type =
  using tGraph = G;
  using tNode = N;
  using tEdge = E;

  std::unordered_set<N> get_nodes(G const &) const;
  std::unordered_set<E> get_incoming_edges(G const &, N const &) const;
  std::unordered_set<E> get_outgoing_edges(G const &, N const &) const;
  N get_src(G const &, E const &) const;
  N get_dst(G const &, E const &) const;
};
*/

template <typename T>
struct BasicGraph {
  using N = T;
  using E = std::pair<N, N>;

  std::unordered_set<T> nodes;
  std::unordered_map<T, std::unordered_set<E>> in_edges, out_edges;

  BasicGraph() : BasicGraph({}, {}) {}

  BasicGraph(std::unordered_set<T> const &nodes, std::unordered_set<E> edges)
      : nodes(), in_edges(), out_edges() {
    this->add_nodes(nodes);
    this->add_edges(edges);
  }

  void add_edge(N const &src, N const &dst) {
    nodes.insert(src);
    nodes.insert(dst);
    out_edges[src].insert({src, dst});
    in_edges[dst].insert({src, dst});
  }

  void add_edge(E const &e) {
    nodes.insert(e.first);
    nodes.insert(e.second);
    out_edges[e.first].insert(e);
    in_edges[e.second].insert(e);
  }

  bool has_edge(N const &src, N const &dst) const {
    auto iter = this->in_edges.find(dst);
    if (iter == this->in_edges.end()) {
      return false;
    }

    std::unordered_set<E> const &dst_in_edges = iter->second;
    return dst_in_edges.find({src, dst}) != dst_in_edges.end();
  }

  bool has_edge(E const &e) const {
    return this->has_edge(e.first, e.second);
  }

  void remove_edge(N const &src, N const &dst) {
    out_edges[src].erase({src, dst});
    in_edges[dst].erase({src, dst});
  }

  void remove_edge(E const &e) {
    out_edges[e.first].erase(e);
    in_edges[e.second].erase(e);
  }

  void add_node(N const &n) {
    nodes.insert(n);
  }

  template <typename Container = std::vector<N>>
  void add_nodes(Container const &nodes) {
    for (auto const &n : nodes) {
      this->add_node(n);
    }
  }

  template <typename Container = std::vector<E>>
  void add_edges(Container const &edges) {
    for (auto const &e : edges) {
      this->add_edge(e);
    }
  }

  bool operator==(BasicGraph<T> const &other) const {
    return this->nodes == other.nodes && this->in_edges == other.in_edges &&
           this->out_edges == other.out_edges;
  }
};

template <typename T>
struct GraphStructure<BasicGraph<T>> {
  using graph_type = BasicGraph<T>;
  using vertex_type = T;
  using edge_type = std::pair<T, T>;

  std::unordered_set<vertex_type> get_nodes(graph_type const &g) const {
    std::unordered_set<vertex_type> nodes(g.nodes);
    return nodes;
  }

  std::unordered_set<edge_type> get_incoming_edges(graph_type const &g,
                                                   vertex_type const &n) const {
    std::unordered_set<edge_type> edges;
    if (g.in_edges.find(n) != g.in_edges.end()) {
      edges.insert(g.in_edges.at(n).begin(), g.in_edges.at(n).end());
    }
    return edges;
  }

  std::unordered_set<edge_type> get_outgoing_edges(graph_type const &g,
                                                   vertex_type const &n) const {
    std::unordered_set<edge_type> edges;
    if (g.out_edges.find(n) != g.out_edges.end()) {
      edges.insert(g.out_edges.at(n).begin(), g.out_edges.at(n).end());
    }
    return edges;
  }

  vertex_type get_src(graph_type const &g, edge_type const &e) const {
    return e.first;
  }

  vertex_type get_dst(graph_type const &g, edge_type const &e) const {
    return e.second;
  }

  void set_src(graph_type const &g, edge_type &e, vertex_type const &n) const {
    e.first = n;
  }

  void set_dst(graph_type const &g, edge_type &e, vertex_type const &n) const {
    e.second = n;
  }
};

} // namespace Utils
} // namespace PCG
} // namespace FlexFlow

#endif // _BASIC_GRAPH_H

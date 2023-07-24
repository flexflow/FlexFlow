#ifndef _GRAPH_STRUCTURES_H
#define _GRAPH_STRUCTURES_H

#include "basic_graph.h"

namespace FlexFlow {
namespace PCG {
namespace Utils {

template <typename BaseStructure>
struct ReverseStructure {
  using graph_type = typename BaseStructure::graph_type;
  using G = graph_type;
  using vertex_type = typename BaseStructure::vertex_type;
  using edge_type = typename BaseStructure::edge_type;

  std::unordered_set<vertex_type> get_nodes(G const &g) const {
    return this->base.get_nodes(g);
  }

  std::unordered_set<edge_type> get_incoming_edges(G const &g,
                                                   vertex_type const &n) const {
    return this->base.get_outgoing_edges(g, n);
  }

  std::unordered_set<edge_type> get_outgoing_edges(G const &g,
                                                   vertex_type const &n) const {
    return this->base.get_incoming_edges(g, n);
  }

  vertex_type get_src(G const &g, edge_type const &e) const {
    return this->base.get_dst(g, e);
  }

  vertex_type get_dst(G const &g, edge_type const &e) const {
    return this->base.get_src(g, e);
  }

  void set_src(G const &g, edge_type &e, vertex_type const &n) const {
    this->base.set_dst(g, e, n);
  }

  void set_dst(G const &g, edge_type &e, vertex_type const &n) const {
    this->base.set_src(g, e, n);
  }

  BaseStructure base;
};

template <typename NotReversed, typename Reversed>
struct UndirectedEdge {
  union Edge {
    NotReversed not_reversed;
    Reversed reversed;

    Edge() {}
  };

  bool is_reversed;
  Edge edge;

  UndirectedEdge() {}

  bool operator==(UndirectedEdge<NotReversed, Reversed> const &other) const {
    if (other.is_reversed != this->is_reversed) {
      return false;
    }
    if (this->is_reversed) {
      return this->edge.reversed == other.edge.reversed;
    } else {
      return this->edge.not_reversed == other.edge.not_reversed;
    }
  }
};

template <typename G, typename BaseStructure = GraphStructure<G>>
struct UndirectedStructure {
  using graph_type = typename BaseStructure::graph_type;
  using vertex_type = typename BaseStructure::vertex_type;
  using not_reversed_edge_type = typename BaseStructure::edge_type;
  using reversed_edge_type =
      typename ReverseStructure<BaseStructure>::edge_type;
  using edge_type = UndirectedEdge<not_reversed_edge_type, reversed_edge_type>;

  std::unordered_set<vertex_type> get_nodes(G const &g) const {
    return this->base.get_nodes(g);
  }

  std::unordered_set<edge_type> get_incoming_edges(G const &g,
                                                   vertex_type const &n) const {
    std::unordered_set<edge_type> incoming;
    auto base_edges = this->base.get_incoming_edges(g, n);
    auto reversed_edges = this->reversed.get_incoming_edges(g, n);

    for (auto const &e : base_edges) {
      edge_type lifted;
      lifted.is_reversed = false;
      lifted.edge.not_reversed = e;
      incoming.insert(lifted);
    }

    for (auto const &e : reversed_edges) {
      edge_type lifted;
      lifted.is_reversed = true;
      lifted.edge.reversed = e;
      incoming.insert(lifted);
    }

    return incoming;
  }

  std::unordered_set<edge_type> get_outgoing_edges(G const &g,
                                                   vertex_type const &n) const {
    std::unordered_set<edge_type> outgoing;
    auto base_edges = this->base.get_outgoing_edges(g, n);
    auto reversed_edges = this->reversed.get_outgoing_edges(g, n);

    for (auto const &e : base_edges) {
      edge_type lifted;
      lifted.is_reversed = false;
      lifted.edge.not_reversed = e;
      outgoing.insert(lifted);
    }

    for (auto const &e : reversed_edges) {
      edge_type lifted;
      lifted.is_reversed = true;
      lifted.edge.reversed = e;
      outgoing.insert(lifted);
    }

    return outgoing;
  }

  vertex_type get_src(G const &g, edge_type const &e) const {
    if (e.is_reversed) {
      return this->reversed.get_src(g, e.edge.reversed);
    } else {
      return this->base.get_src(g, e.edge.not_reversed);
    }
  }

  vertex_type get_dst(G const &g, edge_type const &e) const {
    if (e.is_reversed) {
      return this->reversed.get_dst(g, e.edge.reversed);
    } else {
      return this->base.get_dst(g, e.edge.not_reversed);
    }
  }

  void set_src(G const &g, edge_type &e, vertex_type const &n) const {
    if (e.is_reversed) {
      this->reversed.set_src(g, e.edge.reversed, n);
    } else {
      this->base.set_src(g, e.edge.not_reversed, n);
    }
  }

  void set_dst(G const &g, edge_type &e, vertex_type const &n) const {
    if (e.is_reversed) {
      this->reversed.set_src(g, e.edge.reversed, n);
    } else {
      this->base.set_src(g, e.edge.not_reversed, n);
    }
  }

  BaseStructure base;
  ReverseStructure<BaseStructure> reversed;
};

template <typename G, typename Structure = GraphStructure<G>>
struct invalid_node;

template <typename G,
          typename BaseStructure = GraphStructure<G>,
          typename Invalid = invalid_node<G, BaseStructure>>
struct MultisourceGraphStructure {
  using graph_type = typename BaseStructure::graph_type;
  using vertex_type = typename BaseStructure::vertex_type;
  using edge_type = typename BaseStructure::edge_type;

  std::unordered_set<vertex_type> get_nodes(G const &g) const {
    Invalid invalid;

    std::unordered_set<vertex_type> nodes = this->base.get_nodes(g);
    nodes.insert(invalid());
    return nodes;
  }

  std::unordered_set<edge_type> get_incoming_edges(G const &g,
                                                   vertex_type const &n) const {
    Invalid invalid;

    if (n == invalid()) {
      return {};
    }

    std::unordered_set<edge_type> edges = this->base.get_incoming_edges(g, n);
    if (edges.empty()) {
      edge_type e;
      this->base.set_src(g, e, invalid());
      this->base.set_dst(g, e, n);
      return {e};
    }

    return edges;
  }

  std::unordered_set<edge_type> get_outgoing_edges(G const &g,
                                                   vertex_type const &n) const {
    Invalid invalid;

    if (n == invalid()) {
      std::unordered_set<edge_type> edges;
      for (auto const &node : this->base.get_nodes(g)) {
        if (this->base.get_incoming_edges(g, node).empty()) {
          edge_type e;
          this->base.set_src(g, e, invalid());
          this->base.set_dst(g, e, node);
          edges.insert(e);
        }
      }
      return edges;
    }

    return this->base.get_outgoing_edges(g, n);
  }

  vertex_type get_src(G const &g, edge_type const &e) const {
    return this->base.get_src(g, e);
  }

  vertex_type get_dst(G const &g, edge_type const &e) const {
    return this->base.get_dst(g, e);
  }

  void set_src(G const &g, edge_type &e, vertex_type const &n) const {
    this->base.set_src(g, e, n);
  }

  void set_dst(G const &g, edge_type &e, vertex_type const &n) const {
    this->base.set_dst(g, e, n);
  }

  BaseStructure base;
};
} // namespace Utils
} // namespace PCG
} // namespace FlexFlow

namespace std {
using FlexFlow::PCG::Utils::UndirectedEdge;

template <typename NotReversed, typename Reversed>
struct hash<UndirectedEdge<NotReversed, Reversed>> {
  size_t operator()(UndirectedEdge<NotReversed, Reversed> const &e) const {
    size_t result;
    result = std::hash<bool>()(e.is_reversed);
    if (e.is_reversed) {
      hash_combine(result, e.edge.reversed);
    } else {
      hash_combine(result, e.edge.not_reversed);
    }
    return result;
  }
};
} // namespace std

#endif // _GRAPH_STRUCTURES_H

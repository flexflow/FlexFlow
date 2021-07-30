#ifndef _GRAPH_STRUCTURES_H
#define _GRAPH_STRUCTURES_H

#include "flexflow/basic_graph.h"

namespace FlexFlow::PCG::Utils {
  template <typename BaseStructure>
  struct ReverseStructure {
    using graph_type = typename BaseStructure::graph_type;
    using G = graph_type;
    using vertex_type = typename BaseStructure::vertex_type;
    using edge_type = typename BaseStructure::edge_type;

    std::unordered_set<vertex_type> get_nodes(G const &g) const {
      return this->base.get_nodes(g);
    }

    std::unordered_set<edge_type> get_incoming_edges(G const &g, vertex_type const &n) const {
      return this->base.get_outgoing_edges(g, n);
    }

    std::unordered_set<edge_type> get_outgoing_edges(G const &g, vertex_type const &n) const {
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

  template <
    typename G,
    typename Structure = GraphStructure<G>
  >
  struct invalid_node;

  template <
    typename G,
    typename BaseStructure = GraphStructure<G>,
    typename Invalid = invalid_node<G, BaseStructure>
  >
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

    std::unordered_set<edge_type> get_incoming_edges(G const &g, vertex_type const &n) const {
      Invalid invalid;

      std::unordered_set<edge_type> edges = this->base.get_incoming_edges(g, n);
      if (edges.empty()) {
        edge_type e;
        this->base.set_src(g, e, invalid());
        this->base.set_dst(g, e, n);
        return {e};
      }

      return edges;
    }

    std::unordered_set<edge_type> get_outgoing_edges(G const &g, vertex_type const &n) const {
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
}

#endif // _GRAPH_STRUCTURES_H

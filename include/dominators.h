#ifndef _DOMINATORS_H
#define _DOMINATORS_H

#include <queue>
#include <unordered_set>
#include <unordered_map>

namespace flexflow {
  namespace dominators {
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

    template <typename G, typename BaseStructure = GraphStructure<G>>
    struct ReverseStructure {
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

    template <typename G, typename Structure = GraphStructure<G>>
    void successors(
        G const &g,
        typename Structure::vertex_type const &node,
        std::unordered_set<typename Structure::vertex_type> *succ
    ) {
      Structure s;
      for (auto const &edge : s.get_outgoing_edges(g, node)) {
        succ->insert(s.get_dst(g, edge));
      }
    }

    template <typename G, typename Structure = GraphStructure<G>>
    void predecessors(
        G const &g,
        typename Structure::vertex_type const &node,
        std::unordered_set<typename Structure::vertex_type> *pred
    ) {
      Structure s;
      for (auto const &edge : s.get_incoming_edges(g, node)) {
        pred->insert(s.get_src(g, edge));
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
    void topo_sort(
        G const &g,
        std::vector<typename Structure::vertex_type> *ordering
    ) {
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

      for (auto it = predecessors.begin(); it != predecessors.end(); ) {
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
    std::unordered_map<typename Structure::vertex_type, std::unordered_set<typename Structure::vertex_type>> dominators(G const &g) {
      using N = typename Structure::vertex_type;
      //using E = typename Structure::edge_type;

      //Structure s;

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
    std::unordered_map<typename Structure::vertex_type, std::unordered_set<typename Structure::vertex_type>> post_dominators(G const &g) {
      return dominators<G, ReverseStructure<G, Structure>>(g);
    }

    template <typename G, typename Structure = GraphStructure<G>>
    std::unordered_map<typename Structure::vertex_type, typename Structure::vertex_type> imm_dominators(G const &g) {
      using N = typename Structure::vertex_type;
      //using E = typename Structure::edge_type;

      std::vector<N> topo;
      topo_sort<G, Structure>(g, &topo);
      std::unordered_map<N, int> topo_rank;
      for (int i = 0; i < (int)topo.size(); i++) {
        topo_rank[topo[i]] = i;
      }
      std::unordered_map<N, std::unordered_set<N>> dom = dominators<G, Structure>(g);

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
    std::unordered_map<typename Structure::vertex_type, typename Structure::vertex_type> imm_post_dominators(G const &g) {
      return imm_dominators<G, ReverseStructure<G, Structure>>(g);
    }
  }
}

#endif // _DOMINATORS_H

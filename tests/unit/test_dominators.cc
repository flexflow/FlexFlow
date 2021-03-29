#include "gtest/gtest.h"
#include "dominators.h"
#include "hash_utils.h"
#include "graph.h"

using namespace flexflow::dominators;

template <typename T>
struct BasicGraph {
  using N = T;
  using E = std::pair<N, N>;

  std::unordered_set<T> nodes;
  std::unordered_map<T, std::unordered_set<E>> in_edges, out_edges;

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

  void add_node(N const &n) {
    nodes.insert(n);
  }

  void add_nodes(std::vector<N> const &nodes) {
    for (auto const &n : nodes) {
      this->add_node(n);
    }
  }

  void add_edges(std::vector<E> const &edges) {
    for (auto const &e : edges) {
      this->add_edge(e);
    }
  }
};

namespace flexflow::dominators {
  template <typename T>
  struct GraphStructure<BasicGraph<T>> {
    using graph_type = BasicGraph<T>;
    using vertex_type = T;
    using edge_type = std::pair<T, T>;

    std::unordered_set<vertex_type> get_nodes(graph_type const &g) const {
      std::unordered_set<vertex_type> nodes(g.nodes);
      return nodes;
    }

    std::unordered_set<edge_type> get_incoming_edges(graph_type const &g, vertex_type const &n) const {
      std::unordered_set<edge_type> edges;
      if (g.in_edges.find(n) != g.in_edges.end()) {
        edges.insert(g.in_edges.at(n).begin(), g.in_edges.at(n).end());
      }
      return edges;
    }

    std::unordered_set<edge_type> get_outgoing_edges(graph_type const &g, vertex_type const &n) const {
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

  template <>
  struct invalid_node<::BasicGraph<int>, GraphStructure<::BasicGraph<int>>> {
    int operator()() const {
      return -1;
    }
  };
}

TEST(pred_succ_cessors, basic) {
  BasicGraph<int> g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);

  g.add_edge(0, 2);
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_edge(2, 4);

  using AnswerMap = std::unordered_map<int, std::unordered_set<int>>;

  AnswerMap expected_predecessors;

  expected_predecessors = {
    {0, { }},
    {1, { }},
    {2, {0, 1}},
    {3, {2}},
    {4, {2}}
  };

  AnswerMap expected_successors = {
    {0, {2}},
    {1, {2}},
    {2, {3, 4}},
    {3, { }},
    {4, { }}
  };

  std::unordered_set<int> answer;
  for (auto const &kv : expected_predecessors) {
    answer.clear();
    predecessors<BasicGraph<int>>(g, kv.first, &answer);
    EXPECT_EQ(kv.second, answer) << "^^^ Predecessors for node " << kv.first << std::endl;
  }
  for (auto const &kv : expected_successors) {
    answer.clear();
    successors<BasicGraph<int>>(g, kv.first, &answer);
    EXPECT_EQ(kv.second, answer) << "^^^ Successors for node " << kv.first << std::endl;
  }
}

TEST(topo_sort, basic) {
  BasicGraph<int> g;
  g.add_nodes({0, 1, 2, 3});
  g.add_edges({
    {3, 1},
    {3, 0},
    {1, 0},
    {0, 2}
  });

  std::vector<int> topo_answer = { 3, 1, 0, 2 };

  std::vector<int> topo_result;
  topo_sort(g, &topo_result);
  EXPECT_EQ(topo_result, topo_answer);
}

BasicGraph<int> get_dominator_test_graph() {
  BasicGraph<int> g;
  g.add_nodes({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  g.add_edges({
    {1, 2},
    {1, 7},
    {2, 3},
    {2, 4},
    {3, 6},
    {4, 5},
    {4, 6},
    {5, 6},
    {6, 8},
    {7, 8},
    {8, 9},
    {8, 10},
    {9, 11},
    {10, 11}
  });

  return g;
}

TEST(dominators, basic) {
  BasicGraph<int> g = get_dominator_test_graph();

  std::unordered_map<int, std::unordered_set<int>> answer = {
    {1, {1}},
    {2, {1, 2}},
    {3, {1, 2, 3}},
    {4, {1, 2, 4}},
    {5, {1, 2, 4, 5}},
    {6, {1, 2, 6}},
    {7, {1, 7}},
    {8, {1, 8}},
    {9, {1, 8, 9}},
    {10, {1, 8, 10}},
    {11, {1, 8, 11}}
  };

  EXPECT_EQ(dominators(g), answer);
}

TEST(post_dominators, basic) {
  BasicGraph<int> g = get_dominator_test_graph();

  std::unordered_map<int, std::unordered_set<int>> answer = {
    {1, {1, 8, 11}},
    {2, {2, 6, 8, 11}},
    {3, {3, 6, 8, 11}},
    {4, {4, 6, 8, 11}},
    {5, {5, 6, 8, 11}},
    {6, {6, 8, 11}},
    {7, {7, 8, 11}},
    {8, {8, 11}},
    {9, {9, 11}},
    {10, {10, 11}},
    {11, {11}}
  };

  EXPECT_EQ(post_dominators(g), answer);
}

TEST(imm_dominators, basic) {
  BasicGraph<int> g = get_dominator_test_graph();

  std::unordered_map<int, int> answer = {
    {1, 1}, // no immediate dominator
    {2, 1},
    {3, 2},
    {4, 2},
    {5, 4},
    {6, 2},
    {7, 1},
    {8, 1},
    {9, 8},
    {10, 8},
    {11, 8}
  };

  EXPECT_EQ(imm_dominators(g), answer);
}

TEST(imm_post_dominators, basic) {
  BasicGraph<int> g = get_dominator_test_graph();

  std::unordered_map<int, int> answer = {
    {1, 8},
    {2, 6},
    {3, 6},
    {4, 6},
    {5, 6},
    {6, 8},
    {7, 8},
    {8, 11},
    {9, 11},
    {10, 11},
    {11, 11} // no immediate post dominator
  };

  EXPECT_EQ(imm_post_dominators(g), answer);
}

TEST(imm_post_dominators, multisource) {
  BasicGraph<int> g;

  g.add_nodes({1, 2, 3, 4, 5});
  g.add_edges({
    {1, 3},
    {2, 3},
    {3, 4},
    {3, 5}
  });

  std::unordered_map<int, int> answer = {
    {-1, 3},
    {1, 3},
    {2, 3},
    {3, 3},
    {4, 4},
    {5, 5}
  };

  auto result = imm_post_dominators<decltype(g), MultisourceGraphStructure<decltype(g)>>(g);
  EXPECT_EQ(result, answer);
}

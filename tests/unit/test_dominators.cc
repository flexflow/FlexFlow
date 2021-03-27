#include "gtest/gtest.h"
#include "dominators.h"
#include "hash_utils.h"

using namespace flexflow::dominators;

struct BasicGraph {

  using Node = int;
  using Edge = std::pair<Node, Node>;

  std::unordered_set<int> nodes;
  std::unordered_map<int, std::unordered_set<Edge>> in_edges, out_edges;

  void add_edge(Node const &src, Node const &dst) {
    nodes.insert(src);
    nodes.insert(dst);
    out_edges[src].insert({src, dst});
    in_edges[dst].insert({src, dst});
  }

  void add_edge(Edge const &e) {
    nodes.insert(e.first);
    nodes.insert(e.second);
    out_edges[e.first].insert(e);
    in_edges[e.second].insert(e);
  }

  void add_node(Node const &n) {
    nodes.insert(n);
  }

  void add_nodes(std::vector<Node> const &nodes) {
    for (auto const &n : nodes) {
      this->add_node(n);
    }
  }

  void add_edges(std::vector<Edge> const &edges) {
    for (auto const &e : edges) {
      this->add_edge(e);
    }
  }
};

namespace flexflow::dominators {
  template <>
  struct GraphStructure<BasicGraph> {
    using N = typename BasicGraph::Node;
    using E = typename BasicGraph::Edge;

    std::unordered_set<N> get_nodes(BasicGraph const &g) const {
      std::unordered_set<N> nodes(g.nodes);
      return nodes;
    }

    std::unordered_set<E> get_incoming_edges(BasicGraph const &g, N const &n) const {
      std::unordered_set<E> edges;
      if (g.in_edges.find(n) != g.in_edges.end()) {
        edges.insert(g.in_edges.at(n).begin(), g.in_edges.at(n).end());
      }
      return edges;
    }

    std::unordered_set<E> get_outgoing_edges(BasicGraph const &g, N const &n) const {
      std::unordered_set<E> edges;
      if (g.out_edges.find(n) != g.out_edges.end()) {
        edges.insert(g.out_edges.at(n).begin(), g.out_edges.at(n).end());
      }
      return edges;
    }

    N get_src(BasicGraph const &g, E const &e) const {
      return e.first;
    }

    N get_dst(BasicGraph const &g, E const &e) const {
      return e.second;
    }
  };
}

TEST(pred_succ_cessors, basic) {
  BasicGraph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);

  g.add_edge(0, 2);
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_edge(2, 4);

  using AnswerMap = std::unordered_map<BasicGraph::Node, std::unordered_set<BasicGraph::Node>>;

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
    predecessors<BasicGraph>(g, kv.first, &answer);
    EXPECT_EQ(kv.second, answer) << "^^^ Predecessors for node " << kv.first << std::endl;
  }
  for (auto const &kv : expected_successors) {
    answer.clear();
    successors<BasicGraph>(g, kv.first, &answer);
    EXPECT_EQ(kv.second, answer) << "^^^ Successors for node " << kv.first << std::endl;
  }
}

TEST(topo_sort, basic) {
  BasicGraph g;
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

BasicGraph get_dominator_test_graph() {
  BasicGraph g;
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
  BasicGraph g = get_dominator_test_graph();

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
  BasicGraph g = get_dominator_test_graph();

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
  BasicGraph g = get_dominator_test_graph();

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
  BasicGraph g = get_dominator_test_graph();

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

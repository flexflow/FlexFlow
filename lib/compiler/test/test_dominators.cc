#include "flexflow/basic_graph.h"
#include "flexflow/dominators.h"
#include "flexflow/utils/hash-utils.h"
#include "gtest/gtest.h"

using namespace FlexFlow::PCG::Utils;

namespace FlexFlow::PCG::Utils {
template <>
struct invalid_node<::BasicGraph<int>, GraphStructure<::BasicGraph<int>>> {
  int operator()() const {
    return -1;
  }
};
} // namespace FlexFlow::PCG::Utils

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

  expected_predecessors = {{0, {}}, {1, {}}, {2, {0, 1}}, {3, {2}}, {4, {2}}};

  AnswerMap expected_successors = {
      {0, {2}}, {1, {2}}, {2, {3, 4}}, {3, {}}, {4, {}}};

  std::unordered_set<int> answer;
  for (auto const &kv : expected_predecessors) {
    answer.clear();
    predecessors<BasicGraph<int>>(g, kv.first, &answer);
    EXPECT_EQ(kv.second, answer)
        << "^^^ Predecessors for node " << kv.first << std::endl;
  }
  for (auto const &kv : expected_successors) {
    answer.clear();
    successors<BasicGraph<int>>(g, kv.first, &answer);
    EXPECT_EQ(kv.second, answer)
        << "^^^ Successors for node " << kv.first << std::endl;
  }
}

TEST(topo_sort, basic) {
  BasicGraph<int> g;
  g.add_nodes({0, 1, 2, 3});
  g.add_edges({{3, 1}, {3, 0}, {1, 0}, {0, 2}});

  std::vector<int> topo_answer = {3, 1, 0, 2};

  std::vector<int> topo_result;
  topo_sort(g, &topo_result);
  EXPECT_EQ(topo_result, topo_answer);
}

BasicGraph<int> get_dominator_test_graph() {
  BasicGraph<int> g;
  g.add_nodes({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  g.add_edges({{1, 2},
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
               {10, 11}});

  return g;
}

TEST(dominators, basic) {
  BasicGraph<int> g = get_dominator_test_graph();

  std::unordered_map<int, std::unordered_set<int>> answer = {{1, {1}},
                                                             {2, {1, 2}},
                                                             {3, {1, 2, 3}},
                                                             {4, {1, 2, 4}},
                                                             {5, {1, 2, 4, 5}},
                                                             {6, {1, 2, 6}},
                                                             {7, {1, 7}},
                                                             {8, {1, 8}},
                                                             {9, {1, 8, 9}},
                                                             {10, {1, 8, 10}},
                                                             {11, {1, 8, 11}}};

  EXPECT_EQ(dominators(g), answer);
}

TEST(post_dominators, basic) {
  BasicGraph<int> g = get_dominator_test_graph();

  std::unordered_map<int, std::unordered_set<int>> answer = {{1, {1, 8, 11}},
                                                             {2, {2, 6, 8, 11}},
                                                             {3, {3, 6, 8, 11}},
                                                             {4, {4, 6, 8, 11}},
                                                             {5, {5, 6, 8, 11}},
                                                             {6, {6, 8, 11}},
                                                             {7, {7, 8, 11}},
                                                             {8, {8, 11}},
                                                             {9, {9, 11}},
                                                             {10, {10, 11}},
                                                             {11, {11}}};

  EXPECT_EQ(post_dominators(g), answer);
}

TEST(imm_dominators, basic) {
  BasicGraph<int> g = get_dominator_test_graph();

  std::unordered_map<int, int> answer = {{1, 1}, // no immediate dominator
                                         {2, 1},
                                         {3, 2},
                                         {4, 2},
                                         {5, 4},
                                         {6, 2},
                                         {7, 1},
                                         {8, 1},
                                         {9, 8},
                                         {10, 8},
                                         {11, 8}};

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
      {11, 11} // no immediate post
               // dominator
  };

  EXPECT_EQ(imm_post_dominators(g), answer);
}

TEST(imm_post_dominators, multisource) {
  BasicGraph<int> g;

  g.add_nodes({1, 2, 3, 4, 5});
  g.add_edges({{1, 3}, {2, 3}, {3, 4}, {3, 5}});

  std::unordered_map<int, int> answer = {
      {-1, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 4}, {5, 5}};

  auto result =
      imm_post_dominators<decltype(g), MultisourceGraphStructure<decltype(g)>>(
          g);
  EXPECT_EQ(result, answer);
}

TEST(transitive_reduction, basic) {
  BasicGraph<int> g({1, 2, 3}, {{1, 2}, {2, 3}, {1, 3}});

  BasicGraph<int> answer({1, 2, 3}, {{1, 2}, {2, 3}});

  auto result = transitive_reduction(g);

  EXPECT_EQ(result, answer);
}

TEST(transitive_reduction, medium) {
  BasicGraph<int> g({1, 2, 3, 4, 5, 6, 7},
                    {
                        {1, 4},
                        {1, 5},
                        {2, 3},
                        {2, 4},
                        {2, 6},
                        {3, 4},
                        {4, 5},
                        {4, 6},
                        {5, 6},
                    });

  BasicGraph<int> answer({1, 2, 3, 4, 5, 6, 7},
                         {
                             {1, 4},
                             {2, 3},
                             {3, 4},
                             {4, 5},
                             {5, 6},
                         });

  auto result = transitive_reduction(g);

  EXPECT_EQ(result, answer);
}

TEST(inplace_transitive_reduction, basic) {
  BasicGraph<int> g({1, 2, 3, 4, 5, 6, 7},
                    {
                        {1, 4},
                        {1, 5},
                        {2, 3},
                        {2, 4},
                        {2, 6},
                        {3, 4},
                        {4, 5},
                        {4, 6},
                        {5, 6},
                    });

  BasicGraph<int> answer({1, 2, 3, 4, 5, 6, 7},
                         {
                             {1, 4},
                             {2, 3},
                             {3, 4},
                             {4, 5},
                             {5, 6},
                         });

  inplace_transitive_reduction(g);

  EXPECT_EQ(g, answer);
}

TEST(roots, basic) {
  BasicGraph<int> g({1, 2, 3, 4, 5, 6},
                    {
                        {1, 3},
                        {2, 3},
                        {3, 4},
                        {3, 5},
                        {3, 6},
                    });

  std::unordered_set<int> answer{1, 2};

  auto result = roots(g);

  EXPECT_EQ(result, answer);
}

TEST(leaves, basic) {
  BasicGraph<int> g({1, 2, 3, 4, 5, 6},
                    {{1, 3}, {2, 3}, {3, 4}, {3, 5}, {3, 6}});

  std::unordered_set<int> answer{4, 5, 6};

  auto result = leaves(g);

  EXPECT_EQ(result, answer);
}

TEST(descendants, directed) {
  BasicGraph<int> g({1, 2, 3, 4, 5, 6},
                    {{1, 2}, {2, 3}, {2, 4}, {3, 5}, {4, 5}});

  std::unordered_set<int> answer{2, 3, 4, 5};

  auto result = descendants(g, 2);

  EXPECT_EQ(result, answer);
}

TEST(descendants, undirected) {
  BasicGraph<int> g({1, 2, 3, 4, 5, 6},
                    {{1, 2}, {2, 3}, {2, 4}, {3, 5}, {4, 5}});

  std::unordered_set<int> answer{1, 2, 3, 4, 5};

  auto result =
      descendants<decltype(g), UndirectedStructure<decltype(g)>>(g, 2);

  EXPECT_EQ(result, answer);
}

TEST(weakly_connected_components, basic) {
  BasicGraph<int> g({1, 2, 3, 4, 5, 6}, {{1, 3}, {2, 3}, {4, 5}, {5, 4}});

  std::unordered_set<int> component1{1, 2, 3};
  std::unordered_set<int> component2{4, 5};
  std::unordered_set<int> component3{6};
  auto result = weakly_connected_components(g);

  EXPECT_EQ(result.size(), 3);
  bool component1_found = false;
  bool component2_found = false;
  bool component3_found = false;
  for (std::unordered_set<int> &component : result) {
    if (component.size() == component1.size()) {
      component1_found = true;
      EXPECT_EQ(component, component1);
    } else if (component.size() == component2.size()) {
      component2_found = true;
      EXPECT_EQ(component, component2);
    } else if (component.size() == component3.size()) {
      component3_found = true;
      EXPECT_EQ(component, component3);
    }
  }

  EXPECT_TRUE(component1_found);
  EXPECT_TRUE(component2_found);
  EXPECT_TRUE(component3_found);
}

#include "pcg/dataflow_graph/dataflow_graph.h"
#include "test/utils/doctest.h"
#include "utils/fmt/unordered_set.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DataflowGraph") {
    DataflowGraph<int, std::string> g;

    int n1_label = 1;
    int n2_label = 2;
    int n3_label = 3;
    int n4_label = 4;

    std::string o1_label = "o1";
    std::string o2_label = "o2";
    std::string o3_label = "o3";
    std::string o4_label = "o4";

    OperatorAddedResult n1_added = g.add_operator(n1_label, {}, {o1_label});
    Node n1 = n1_added.node;
    MultiDiOutput o1 = get_only(n1_added.outputs);

    OperatorAddedResult n2_added = g.add_operator(n2_label, {}, {o2_label});
    Node n2 = n2_added.node;
    MultiDiOutput o2 = get_only(n2_added.outputs);

    OperatorAddedResult n3_added = g.add_operator(n3_label, {}, {o3_label});
    Node n3 = n3_added.node;
    MultiDiOutput o3 = get_only(n3_added.outputs);

    OperatorAddedResult n4_added =
        g.add_operator(n4_label, {o1, o2, o3}, {o4_label});
    Node n4 = n4_added.node;
    MultiDiOutput o4 = get_only(n4_added.outputs);

    SUBCASE("get_inputs") {
      std::vector<MultiDiOutput> result = get_inputs(g, n4);
      std::vector<MultiDiOutput> correct = {o1, o2, o3};
      CHECK(result == correct);
    }

    SUBCASE("get_outputs") {
      std::vector<MultiDiOutput> result = get_outputs(g, n4);
      std::vector<MultiDiOutput> correct = {o4};
      CHECK(result == correct);
    }
  }
}

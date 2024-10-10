#include "utils/dot_file.h"
#include <doctest/doctest.h>
#include <sstream>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DotFile") {
    std::ostringstream oss;
    DotFile<std::string> dotFile(oss);
    SUBCASE("add_node") {
      dotFile.add_node("A", {{"shape", "circle"}, {"label", "Node A"}});
      dotFile.add_node("B", {{"shape", "rectangle"}, {"label", "Node B"}});

      dotFile.close();

      std::string expectedOutput = R"EXPECTED_OUTPUT(digraph taskgraph {
  node0 [label=Node A,shape=circle];
  node1 [label=Node B,shape=rectangle];
})EXPECTED_OUTPUT";

      CHECK(oss.str() == expectedOutput);
    }

    SUBCASE("add_edge") {
      dotFile.add_edge("A", "B");
      dotFile.add_edge("B", "C");

      dotFile.close();

      std::string expectedOutput = R"EXPECTED_OUTPUT(digraph taskgraph {
  node0 -> node1;
  node1 -> node2;
})EXPECTED_OUTPUT";

      CHECK(oss.str() == expectedOutput);
    }

    SUBCASE("add_record_node") {
      RecordFormatter rf;

      rf << "Field1";
      rf << 42;
      rf << "Field2";
      rf << float(3.14);

      dotFile.add_record_node("A", rf);

      dotFile.close();

      std::string expectedOutput =
          R"EXPECTED_OUTPUT(digraph taskgraph {
  node0 [label="{ Field1 | 42 | Field2 | 3.140000e+00 }",shape=record];
})EXPECTED_OUTPUT";

      CHECK(oss.str() == expectedOutput);
    }

    SUBCASE("add_node_to_subgraph") {
      size_t subgraph1 = dotFile.add_subgraph();
      size_t subgraph2 = dotFile.add_subgraph(subgraph1);

      dotFile.add_node_to_subgraph("A", subgraph1);
      dotFile.add_node_to_subgraph("B", subgraph2);

      dotFile.close();

      std::string expectedOutput = R"EXPECTED_OUTPUT(digraph taskgraph {
subgraph cluster_0 {
node1;
node0;
subgraph cluster_1 {
node1;
}
}
})EXPECTED_OUTPUT";

      CHECK(oss.str() == expectedOutput);
    }
  }
}

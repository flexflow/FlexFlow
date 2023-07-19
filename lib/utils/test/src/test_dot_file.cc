#include "doctest.h"
#include <sstream>
#include "utils/dot_file.h"

TEST_CASE("DotFile") {
  SUBCASE("add_node") {
    std::ostringstream oss;
    DotFile<std::string> dotFile(oss);

    dotFile.add_node("A", {{"shape", "circle"}, {"label", "Node A"}});
    dotFile.add_node("B", {{"shape", "rectangle"}, {"label", "Node B"}});

    dotFile.close();

    std::string expectedOutput = "digraph taskgraph {\n  node0 [label=Node A,shape=circle];\n  node1 [label=Node B,shape=rectangle];\n}";

    CHECK_EQ(oss.str(), expectedOutput);
  }

  SUBCASE("add_edge") {
    std::ostringstream oss;
    DotFile<std::string> dotFile(oss);

    dotFile.add_edge("A", "B");
    dotFile.add_edge("B", "C");

    dotFile.close();

    std::string expectedOutput = R"EXPECTED_OUTPUT(digraph taskgraph {
  node0 -> node1;
  node1 -> node2;
})EXPECTED_OUTPUT";

    CHECK_EQ(oss.str(), expectedOutput);
  }

  SUBCASE("add_record_node") {
    std::ostringstream oss;
    DotFile<std::string> dotFile(oss);
    RecordFormatter rf;

    rf << "Field1" ;
    rf<< 42;
    rf << "Field2";
    rf << float(3.14);

    dotFile.add_record_node("A", rf);

    dotFile.close();

    std::string expectedOutput =
        R"EXPECTED_OUTPUT(digraph taskgraph {
  node0 [label="{ Field1 | 42 | Field2 | 3.140000e+00 }",shape=record];
})EXPECTED_OUTPUT";

  

    CHECK_EQ(oss.str(), expectedOutput);
  }

  SUBCASE("add_node_to_subgraph") {
    std::ostringstream oss;
    DotFile<std::string> dotFile(oss);

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

    CHECK_EQ(oss.str(), expectedOutput);
  }
}

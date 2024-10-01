#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_singleton_pattern") {
    OpenDataflowGraph g =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    SUBCASE("0 nodes") {
      UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};

      CHECK_FALSE(is_singleton_pattern(pattern));
    }

    NodeAddedResult n0_added = g.add_node({}, 1);
    OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

    SUBCASE("1 node") {
      UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};

      CHECK(is_singleton_pattern(pattern));
    }

    NodeAddedResult n1_added = g.add_node({v0}, 1);
    OpenDataflowValue v1 = OpenDataflowValue{get_only(n1_added.outputs)};

    SUBCASE("more than 1 node") {
      UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};

      CHECK_FALSE(is_singleton_pattern(pattern));
    }
  }
}

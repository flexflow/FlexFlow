#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ParallelSplit and SerialSplit equality checks") {

    SUBCASE("ParallelSplit::operator== - commutativity") {
      ParallelSplit p1 = ParallelSplit{Node(1), Node(2), Node(3)};
      ParallelSplit p2 = ParallelSplit{Node(2), Node(1), Node(3)};
      ParallelSplit p3 = ParallelSplit{Node(3), Node(2), Node(1)};
      CHECK(p1 == p2);
      CHECK(p2 == p3);
      CHECK(p1 == p3);
    }

    SUBCASE("SerialSplit::operator== - non-commutativity") {
      SerialSplit p1 = SerialSplit{Node(1), Node(2), Node(3)};
      SerialSplit p2 = SerialSplit{Node(2), Node(1), Node(3)};
      SerialSplit p3 = SerialSplit{Node(3), Node(2), Node(1)};
      CHECK(p1 != p2);
      CHECK(p2 != p3);
      CHECK(p1 != p3);
    }

    SUBCASE("operator==, mixed case, nested commutativity") {
      std::vector<Node> n = {Node(0), Node(1), Node(2), Node(3)};

      // All definitions are equivalent, since ParallelSplit commutes
      ParallelSplit p1 = ParallelSplit{
          n.at(3), SerialSplit{ParallelSplit{n.at(2), n.at(1)}, n.at(2)}};
      ParallelSplit p2 = ParallelSplit{
          n.at(3), SerialSplit{ParallelSplit{n.at(1), n.at(2)}, n.at(2)}};
      ParallelSplit p3 = ParallelSplit{
          SerialSplit{ParallelSplit{n.at(1), n.at(2)}, n.at(2)}, n.at(3)};
      ParallelSplit p4 = ParallelSplit{
          SerialSplit{ParallelSplit{n.at(2), n.at(1)}, n.at(2)}, n.at(3)};

      CHECK(p1 == p2);
      CHECK(p1 == p3);
      CHECK(p1 == p4);
      CHECK(p2 == p3);
      CHECK(p2 == p4);
      CHECK(p3 == p4);
    }
  }
}

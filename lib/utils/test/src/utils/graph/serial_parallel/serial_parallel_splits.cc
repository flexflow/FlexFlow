#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ParallelSplit::operator== - base testing") {
    std::vector<Node> n = {
        Node(0), Node(1), Node(2), Node(3), Node(4), Node(5)};

    // The following definitions are equivalent
    ParallelSplit p1{n[0],
                     SerialSplit{n[1], n[2]},
                     SerialSplit{n[3], ParallelSplit{n[4], n[5], n[0]}}};
    ParallelSplit p2{SerialSplit{n[1], n[2]},
                     n[0],
                     SerialSplit{n[3], ParallelSplit{n[4], n[5], n[0]}}};
    ParallelSplit p3{n[0],
                     SerialSplit{n[1], n[2]},
                     SerialSplit{n[3], ParallelSplit{n[0], n[5], n[4]}}};
    ParallelSplit p4{SerialSplit{n[3], ParallelSplit{n[5], n[4], n[0]}},
                     SerialSplit{n[1], n[2]},
                     n[0]};
    std::vector<ParallelSplit> p = {p1, p2, p3, p4};

    SUBCASE("Checking for reciprocal equality") {
      std::vector<std::pair<ParallelSplit, ParallelSplit>> pairs = {
          {p1, p2}, {p1, p3}, {p1, p4}, {p2, p4}, {p3, p4}};
      for (auto const &[pa, pb] : pairs) {
        CHECK(pa == pb);
        CHECK(pb == pa);
        CHECK_FALSE(pa != pb);
        CHECK_FALSE(pb != pa);
      }
    }

    SUBCASE("Checking for not-equality") {
      // Not equivalent to the previous: serial order differs
      ParallelSplit p5{n[0],
                       SerialSplit{n[2], n[1]},
                       SerialSplit{n[3], ParallelSplit{n[4], n[5], n[0]}}};
      ParallelSplit p6{n[0],
                       SerialSplit{n[1], n[2]},
                       SerialSplit{ParallelSplit{n[4], n[5], n[0]}, n[3]}};

      for (auto const &pi : p) {
        CHECK(pi != p5);
        CHECK(pi != p6);
      }
    }
  }

  TEST_CASE("ParallelSplit::operator== - nested SP") {
    std::vector<Node> n = {Node(0), Node(1), Node(2), Node(3)};

    // All definitions are equivalent
    ParallelSplit p1{n[3], SerialSplit{ParallelSplit{n[2], n[1]}, n[2]}};
    ParallelSplit p2{n[3], SerialSplit{ParallelSplit{n[1], n[2]}, n[2]}};
    ParallelSplit p3{SerialSplit{ParallelSplit{n[1], n[2]}, n[2]}, n[3]};
    ParallelSplit p4{SerialSplit{ParallelSplit{n[2], n[1]}, n[2]}, n[3]};

    std::vector<std::pair<ParallelSplit, ParallelSplit>> pairs = {
        {p1, p2}, {p1, p3}, {p1, p4}, {p2, p4}, {p3, p4}};
    for (auto const &[pa, pb] : pairs) {
      CHECK(pa == pb);
      CHECK(pb == pa);
      CHECK_FALSE(pa != pb);
      CHECK_FALSE(pb != pa);
    }
  }
}

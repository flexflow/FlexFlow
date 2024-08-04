#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_isempty.h"
#include "utils/graph/serial_parallel/serial_parallel_normalize.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  // TODO: reorganize the tests and extend the existing ones

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
    std::vector<ParallelSplit> p = {p1, p2, p3, p4};

    std::vector<std::pair<ParallelSplit, ParallelSplit>> pairs = {
        {p1, p2}, {p1, p3}, {p1, p4}, {p2, p4}, {p3, p4}};
    for (auto const &[pa, pb] : pairs) {
      CHECK(pa == pb);
      CHECK(pb == pa);
      CHECK_FALSE(pa != pb);
      CHECK_FALSE(pb != pa);
    }
  }

  TEST_CASE("isempty function") {
    Node n1{1};
    Node n2{2};

    SerialParallelDecomposition node_decomp = SerialParallelDecomposition{n1};
    SerialParallelDecomposition empty_serial =
        SerialParallelDecomposition{SerialSplit{}};
    SerialParallelDecomposition empty_parallel =
        SerialParallelDecomposition{ParallelSplit{}};
    SerialParallelDecomposition serial_with_node =
        SerialParallelDecomposition{SerialSplit{n1}};
    SerialParallelDecomposition parallel_with_node =
        SerialParallelDecomposition{ParallelSplit{n1}};
    SerialParallelDecomposition nested_serial =
        SerialParallelDecomposition{SerialSplit{ParallelSplit{}}};
    SerialParallelDecomposition nested_parallel =
        SerialParallelDecomposition{ParallelSplit{SerialSplit{}}};
    ParallelSplit deeply_nested =
        ParallelSplit{SerialSplit{ParallelSplit{SerialSplit{}}}};
    SerialSplit sparse =
        SerialSplit{ParallelSplit{}, ParallelSplit{SerialSplit{}}};
    SerialSplit sparse_with_node =
        SerialSplit{ParallelSplit{}, ParallelSplit{SerialSplit{}, n2}};

    CHECK_FALSE(isempty(node_decomp));
    CHECK(isempty(empty_serial));
    CHECK(isempty(empty_parallel));
    CHECK_FALSE(isempty(serial_with_node));
    CHECK_FALSE(isempty(parallel_with_node));
    CHECK(isempty(nested_serial));
    CHECK(isempty(nested_parallel));
    CHECK(isempty(deeply_nested));
    CHECK(isempty(sparse));
    CHECK_FALSE(isempty(sparse_with_node));
  }

  TEST_CASE("normalize function") {
    Node n1{1};
    Node n2{2};
    Node n3{3};

    SerialParallelDecomposition node_decomp = SerialParallelDecomposition{n1};
    SerialParallelDecomposition serial_with_single_node =
        SerialParallelDecomposition{SerialSplit{n1}};
    SerialParallelDecomposition parallel_with_single_node =
        SerialParallelDecomposition{ParallelSplit{n1}};
    SerialParallelDecomposition mixed_serial =
        SerialParallelDecomposition{SerialSplit{ParallelSplit{n1}, n2}};
    SerialParallelDecomposition mixed_parallel =
        SerialParallelDecomposition{ParallelSplit{SerialSplit{n1}, n2}};
    SerialParallelDecomposition nested = SerialParallelDecomposition{
        ParallelSplit{SerialSplit{ParallelSplit{n1, n2}}, n3, SerialSplit{}}};

    CHECK(normalize(serial_with_single_node).get<Node>() ==
          node_decomp.get<Node>());
    CHECK(normalize(parallel_with_single_node).get<Node>() ==
          node_decomp.get<Node>());

    CHECK(normalize(mixed_serial).get<SerialSplit>() == SerialSplit{n1, n2});
    CHECK(normalize(mixed_parallel).get<ParallelSplit>() ==
          ParallelSplit{n1, n2});
    CHECK(normalize(nested).get<ParallelSplit>() == ParallelSplit{n1, n2, n3});
  }
}

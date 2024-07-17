#include "test/utils/doctest.h"
#include "utils/graph/serialparallel.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("Parallel::operator== - base testing") {
    std::vector<Node> n = {
        Node(0), Node(1), Node(2), Node(3), Node(4), Node(5)};

    // The following definitions are equivalent
    Parallel p1 = {{n[0],
                    Serial{{n[1], n[2]}},
                    Serial{{n[3], Parallel{{n[4], n[5], n[0]}}}}}};
    Parallel p2 = {{Serial{{n[1], n[2]}},
                    n[0],
                    Serial{{n[3], Parallel{{n[4], n[5], n[0]}}}}}};
    Parallel p3 = {{n[0],
                    Serial{{n[1], n[2]}},
                    Serial{{n[3], Parallel{{n[0], n[5], n[4]}}}}}};
    Parallel p4 = {{Serial{{n[3], Parallel{{n[5], n[4], n[0]}}}},
                    Serial{{n[1], n[2]}},
                    n[0]}};
    std::vector p = {p1, p2, p3, p4};

    SUBCASE("Checking for reciprocal equality") {
      for (auto const &[pa, pb] : pairs(p)) {
        CHECK(pa == pb);
        CHECK(pb == pa);
        CHECK_FALSE(pa != pb);
        CHECK_FALSE(pb != pa);
      }
    }
    SUBCASE("Checking for not-equality") {
      // Not equivalent to the previous: serial order differs
      Parallel p5 = {{n[0],
                      Serial{{n[2], n[1]}},
                      Serial{{n[3], Parallel{{n[4], n[5], n[0]}}}}}};
      Parallel p6 = {{n[0],
                      Serial{{n[1], n[2]}},
                      Serial{{Parallel{{n[4], n[5], n[0]}}, n[3]}}}};

      for (auto const &pi : p) {
        CHECK(pi != p5);
        CHECK(pi != p6);
      }
    }
  }
  TEST_CASE("Parallel::operator== - nested SP") {
    std::vector<Node> n = {Node(0), Node(1), Node(2), Node(3)};

    // All definitions are equivalent
    Parallel p1 = {{n[3], Serial{{Parallel{{n[2], n[1]}}, n[2]}}}};
    Parallel p2 = {{n[3], Serial{{Parallel{{n[1], n[2]}}, n[2]}}}};
    Parallel p3 = {{Serial{{Parallel{{n[1], n[2]}}, n[2]}}, n[3]}};
    Parallel p4 = {{Serial{{Parallel{{n[2], n[1]}}, n[2]}}, n[3]}};
    std::vector p = {p1, p2, p3, p4};

    for (auto const &[pa, pb] : pairs(p)) {
      CHECK(pa == pb);
      CHECK(pb == pa);
      CHECK_FALSE(pa != pb);
      CHECK_FALSE(pb != pa);
    }
  }

  TEST_CASE("isempty function") {
    Node n1{1};
    Node n2{2};

    SerialParallelDecomposition node_decomp = n1;
    SerialParallelDecomposition empty_serial = Serial{{}};
    SerialParallelDecomposition empty_parallel = Parallel{{}};
    SerialParallelDecomposition serial_with_node = Serial{{n1}};
    SerialParallelDecomposition parallel_with_node = Parallel{{n1}};
    SerialParallelDecomposition nested_serial = Serial{{Parallel{}}};
    SerialParallelDecomposition nested_parallel = Parallel{{Serial{}}};
    Parallel deeply_nested = Parallel{{Serial{{Parallel{{Serial{}}}}}}};
    Serial sparse = Serial{{Parallel{}, Parallel{{Serial{}}}}};
    Serial sparse_with_node = Serial{{Parallel{}, Parallel{{Serial{}, n2}}}};

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

    SerialParallelDecomposition node_decomp = n1;
    SerialParallelDecomposition serial_with_single_node = Serial{{n1}};
    SerialParallelDecomposition parallel_with_single_node = Parallel{{n1}};
    SerialParallelDecomposition mixed_serial = Serial{{Parallel{{n1}}, n2}};
    SerialParallelDecomposition mixed_parallel = Parallel{{Serial{{n1}}, n2}};
    SerialParallelDecomposition nested =
        Parallel{{Serial{{Parallel{{n1, n2}}}}, n3, Serial{{}}}};

    CHECK(std::get<Node>(normalize(node_decomp)) ==
          std::get<Node>(node_decomp));
    CHECK(std::get<Node>(normalize(serial_with_single_node)) ==
          std::get<Node>(node_decomp));
    CHECK(std::get<Node>(normalize(parallel_with_single_node)) ==
          std::get<Node>(node_decomp));

    CHECK(std::get<Serial>(normalize(mixed_serial)) == Serial{{n1, n2}});
    CHECK(std::get<Parallel>(normalize(mixed_parallel)) == Parallel{{n1, n2}});
    CHECK(std::get<Parallel>(normalize(nested)) == Parallel{{n1, n2, n3}});
  }
}

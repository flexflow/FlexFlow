#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_leaves") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};

    GenericBinarySPDecompositionTreeImplementation<BinarySPDecompositionTree,
                                                   BinarySeriesSplit,
                                                   BinaryParallelSplit,
                                                   Node>
        impl = generic_impl_for_binary_sp_tree();

    auto generic_get_leaves = [&](BinarySPDecompositionTree const &tree) {
      return get_leaves(tree, impl);
    };

    SUBCASE("leaf") {
      BinarySPDecompositionTree input = BinarySPDecompositionTree{n1};

      std::unordered_multiset<Node> result = generic_get_leaves(input);
      std::unordered_multiset<Node> correct = {n1};

      CHECK(result == correct);
    }

    SUBCASE("series split") {
      SUBCASE("children are not the same") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree{
            BinarySeriesSplit{
                BinarySPDecompositionTree{n1},
                BinarySPDecompositionTree{n2},
            },
        };

        std::unordered_multiset<Node> result = generic_get_leaves(input);
        std::unordered_multiset<Node> correct = {n1, n2};

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree{
            BinarySeriesSplit{
                BinarySPDecompositionTree{n1},
                BinarySPDecompositionTree{n1},
            },
        };

        std::unordered_multiset<Node> result = generic_get_leaves(input);
        std::unordered_multiset<Node> correct = {n1, n1};

        CHECK(result == correct);
      }
    }

    SUBCASE("parallel split") {
      SUBCASE("children are not the same") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree{
            BinaryParallelSplit{
                BinarySPDecompositionTree{n1},
                BinarySPDecompositionTree{n2},
            },
        };

        std::unordered_multiset<Node> result = generic_get_leaves(input);
        std::unordered_multiset<Node> correct = {n1, n2};

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree{
            BinaryParallelSplit{
                BinarySPDecompositionTree{n1},
                BinarySPDecompositionTree{n1},
            },
        };

        std::unordered_multiset<Node> result = generic_get_leaves(input);
        std::unordered_multiset<Node> correct = {n1, n1};

        CHECK(result == correct);
      }
    }

    auto make_series_split = [](BinarySPDecompositionTree const &lhs,
                                BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs,
                                  BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) { return BinarySPDecompositionTree{n}; };

    SUBCASE("nested") {
      BinarySPDecompositionTree input = make_parallel_split(
          make_series_split(make_leaf(n1),
                            make_series_split(make_leaf(n2), make_leaf(n3))),
          make_parallel_split(make_leaf(n2), make_leaf(n1)));

      std::unordered_multiset<Node> result = generic_get_leaves(input);
      std::unordered_multiset<Node> correct = {n1, n1, n2, n2, n3};

      CHECK(result == correct);
    }
  }
}

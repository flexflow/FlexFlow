#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/hash.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<GenericBinarySPDecompositionTree<int>>") {
    SUBCASE("leaf") {
      GenericBinarySPDecompositionTree<int> leaf_5 =
          make_generic_binary_sp_leaf(5);
      size_t leaf_5_hash = get_std_hash(leaf_5);

      SUBCASE("leaves with same labels hash to the same value") {
        GenericBinarySPDecompositionTree<int> also_leaf_5 =
            make_generic_binary_sp_leaf(5);
        size_t also_leaf_5_hash = get_std_hash(also_leaf_5);

        CHECK(leaf_5_hash == also_leaf_5_hash);
      }

      SUBCASE("leaves with different labels hash to different values") {
        GenericBinarySPDecompositionTree<int> leaf_6 =
            make_generic_binary_sp_leaf(6);
        size_t leaf_6_hash = get_std_hash(leaf_6);

        CHECK(leaf_5_hash != leaf_6_hash);
      }
    }

    SUBCASE("series split") {
      GenericBinarySPDecompositionTree<int> series_5_6 =
          make_generic_binary_series_split(make_generic_binary_sp_leaf(5),
                                           make_generic_binary_sp_leaf(6));
      size_t series_5_6_hash = get_std_hash(series_5_6);

      SUBCASE("same children lead to the same hash") {
        GenericBinarySPDecompositionTree<int> also_series_5_6 =
            make_generic_binary_series_split(make_generic_binary_sp_leaf(5),
                                             make_generic_binary_sp_leaf(6));
        size_t also_series_5_6_hash = get_std_hash(also_series_5_6);

        CHECK(series_5_6_hash == also_series_5_6_hash);
      }

      SUBCASE("hash is order dependent") {
        GenericBinarySPDecompositionTree<int> series_6_5 =
            make_generic_binary_series_split(make_generic_binary_sp_leaf(6),
                                             make_generic_binary_sp_leaf(5));
        size_t series_6_5_hash = get_std_hash(series_6_5);

        CHECK(series_5_6_hash != series_6_5_hash);
      }

      SUBCASE("different left child leads to different hash") {
        GenericBinarySPDecompositionTree<int> series_4_6 =
            make_generic_binary_series_split(make_generic_binary_sp_leaf(4),
                                             make_generic_binary_sp_leaf(6));
        size_t series_4_6_hash = get_std_hash(series_4_6);

        CHECK(series_5_6_hash != series_4_6_hash);
      }

      SUBCASE("different right child leads to different hash") {
        GenericBinarySPDecompositionTree<int> series_5_7 =
            make_generic_binary_series_split(make_generic_binary_sp_leaf(5),
                                             make_generic_binary_sp_leaf(7));
        size_t series_5_7_hash = get_std_hash(series_5_7);

        CHECK(series_5_6_hash != series_5_7_hash);
      }
    }

    SUBCASE("parallel split") {
      GenericBinarySPDecompositionTree<int> parallel_5_6 =
          make_generic_binary_parallel_split(make_generic_binary_sp_leaf(5),
                                             make_generic_binary_sp_leaf(6));
      size_t parallel_5_6_hash = get_std_hash(parallel_5_6);

      SUBCASE("same children lead to the same hash") {
        GenericBinarySPDecompositionTree<int> also_parallel_5_6 =
            make_generic_binary_parallel_split(make_generic_binary_sp_leaf(5),
                                               make_generic_binary_sp_leaf(6));
        size_t also_parallel_5_6_hash = get_std_hash(also_parallel_5_6);

        CHECK(parallel_5_6_hash == also_parallel_5_6_hash);
      }

      SUBCASE("hash is order dependent") {
        GenericBinarySPDecompositionTree<int> parallel_6_5 =
            make_generic_binary_parallel_split(make_generic_binary_sp_leaf(6),
                                               make_generic_binary_sp_leaf(5));
        size_t parallel_6_5_hash = get_std_hash(parallel_6_5);

        CHECK(parallel_5_6_hash != parallel_6_5_hash);
      }

      SUBCASE("different left child leads to different hash") {
        GenericBinarySPDecompositionTree<int> parallel_4_6 =
            make_generic_binary_parallel_split(make_generic_binary_sp_leaf(4),
                                               make_generic_binary_sp_leaf(6));
        size_t parallel_4_6_hash = get_std_hash(parallel_4_6);

        CHECK(parallel_5_6_hash != parallel_4_6_hash);
      }

      SUBCASE("different right child leads to different hash") {
        GenericBinarySPDecompositionTree<int> parallel_5_7 =
            make_generic_binary_parallel_split(make_generic_binary_sp_leaf(5),
                                               make_generic_binary_sp_leaf(7));
        size_t parallel_5_7_hash = get_std_hash(parallel_5_7);

        CHECK(parallel_5_6_hash != parallel_5_7_hash);
      }
    }
  }
}

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/json.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/fmt.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("adl_serializer<GenericBinarySPDecompositionTree<T>>") {
    SUBCASE("leaf") {
      GenericBinarySPDecompositionTree<int> tt = make_generic_binary_sp_leaf(5);

      nlohmann::json tt_json = {
        {"__type", "GenericBinarySPDecompositionTree"},
        {"type", "leaf"},
        {"value", 5},
      };

      SUBCASE("to_json") {
        nlohmann::json result = tt;
        nlohmann::json correct = tt_json;

        CHECK(result == correct);
      }

      SUBCASE("from_json") {
        GenericBinarySPDecompositionTree<int> result = tt_json.get<GenericBinarySPDecompositionTree<int>>();
        GenericBinarySPDecompositionTree<int> correct = tt;

        CHECK(result == correct);
      }
    }

    SUBCASE("series split") {
      GenericBinarySPDecompositionTree<int> tt = 
        make_generic_binary_series_split(
          make_generic_binary_sp_leaf(2),
          make_generic_binary_sp_leaf(5));

      nlohmann::json tt_json = {
        {"__type", "GenericBinarySPDecompositionTree"},
        {"type", "series"},
        {
          "value", 
          {
            {"__type", "GenericBinarySeriesSplit"},
            {
              "left_child",
              {
                {"__type", "GenericBinarySPDecompositionTree"},
                {"type", "leaf"},
                {"value", 2},
              },
            },
            {
              "right_child",
              {
                {"__type", "GenericBinarySPDecompositionTree"},
                {"type", "leaf"},
                {"value", 5},
              },
            },
          },
        },
      };

      SUBCASE("to_json") {
        nlohmann::json result = tt;
        nlohmann::json correct = tt_json;

        CHECK(result == correct);
      }

      SUBCASE("from_json") {
        GenericBinarySPDecompositionTree<int> result = tt_json.get<GenericBinarySPDecompositionTree<int>>();
        GenericBinarySPDecompositionTree<int> correct = tt;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallel split") {
      GenericBinarySPDecompositionTree<int> tt = 
        make_generic_binary_parallel_split(
          make_generic_binary_sp_leaf(2),
          make_generic_binary_sp_leaf(5));

      nlohmann::json tt_json = {
        {"__type", "GenericBinarySPDecompositionTree"},
        {"type", "parallel"},
        {
          "value", 
          {
            {"__type", "GenericBinaryParallelSplit"},
            {
              "left_child",
              {
                {"__type", "GenericBinarySPDecompositionTree"},
                {"type", "leaf"},
                {"value", 2},
              },
            },
            {
              "right_child",
              {
                {"__type", "GenericBinarySPDecompositionTree"},
                {"type", "leaf"},
                {"value", 5},
              },
            },
          },
        },
      };

      SUBCASE("to_json") {
        nlohmann::json result = tt;
        nlohmann::json correct = tt_json;

        CHECK(result == correct);
      }

      SUBCASE("from_json") {
        GenericBinarySPDecompositionTree<int> result = tt_json.get<GenericBinarySPDecompositionTree<int>>();
        GenericBinarySPDecompositionTree<int> correct = tt;

        CHECK(result == correct);
      }
    }
  }
}

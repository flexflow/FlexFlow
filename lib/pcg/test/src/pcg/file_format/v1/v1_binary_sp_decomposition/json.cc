#include "pcg/file_format/v1/v1_binary_sp_decomposition/json.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("adl_serializer<V1BinarySPDecomposition>") {
    V1BinarySPDecomposition example_tree = V1BinarySPDecomposition{
        V1BinarySeriesSplit{
            V1BinarySPDecomposition{
                V1BinaryParallelSplit{
                    V1BinarySPDecomposition{2},
                    V1BinarySPDecomposition{2},
                },
            },
            V1BinarySPDecomposition{3},
        },
    };

    nlohmann::json example_json = {
        {"type", "series"},
        {
            "left_child",
            {
                {"type", "parallel"},
                {
                    "left_child",
                    {
                        {"type", "leaf"},
                        {"value", 2},
                    },
                },
                {
                    "right_child",
                    {
                        {"type", "leaf"},
                        {"value", 2},
                    },
                },
            },
        },
        {
            "right_child",
            {
                {"type", "leaf"},
                {"value", 3},
            },
        },
    };

    SUBCASE("to_json") {
      nlohmann::json result = example_tree;
      nlohmann::json correct = example_json;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      V1BinarySPDecomposition result =
          example_json.get<V1BinarySPDecomposition>();
      V1BinarySPDecomposition correct = example_tree;

      CHECK(result == correct);
    }
  }

  TEST_CASE("adl_serializer<V1BinarySeriesSplit>") {
    V1BinarySeriesSplit example_split = V1BinarySeriesSplit{
        V1BinarySPDecomposition{
            V1BinaryParallelSplit{
                V1BinarySPDecomposition{2},
                V1BinarySPDecomposition{2},
            },
        },
        V1BinarySPDecomposition{3},
    };

    nlohmann::json example_json = {
        {
            "left_child",
            {
                {"type", "parallel"},
                {
                    "left_child",
                    {
                        {"type", "leaf"},
                        {"value", 2},
                    },
                },
                {
                    "right_child",
                    {
                        {"type", "leaf"},
                        {"value", 2},
                    },
                },
            },
        },
        {
            "right_child",
            {
                {"type", "leaf"},
                {"value", 3},
            },
        },
    };

    SUBCASE("to_json") {
      nlohmann::json result = example_split;
      nlohmann::json correct = example_json;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      V1BinarySeriesSplit result = example_json.get<V1BinarySeriesSplit>();
      V1BinarySeriesSplit correct = example_split;

      CHECK(result == correct);
    }
  }

  TEST_CASE("adl_serializer<V1BinaryParallelSplit>") {
    V1BinaryParallelSplit example_split = V1BinaryParallelSplit{
        V1BinarySPDecomposition{
            V1BinaryParallelSplit{
                V1BinarySPDecomposition{2},
                V1BinarySPDecomposition{2},
            },
        },
        V1BinarySPDecomposition{3},
    };

    nlohmann::json example_json = {
        {
            "left_child",
            {
                {"type", "parallel"},
                {
                    "left_child",
                    {
                        {"type", "leaf"},
                        {"value", 2},
                    },
                },
                {
                    "right_child",
                    {
                        {"type", "leaf"},
                        {"value", 2},
                    },
                },
            },
        },
        {
            "right_child",
            {
                {"type", "leaf"},
                {"value", 3},
            },
        },
    };

    SUBCASE("to_json") {
      nlohmann::json result = example_split;
      nlohmann::json correct = example_json;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      V1BinaryParallelSplit result = example_json.get<V1BinaryParallelSplit>();
      V1BinaryParallelSplit correct = example_split;

      CHECK(result == correct);
    }
  }
}

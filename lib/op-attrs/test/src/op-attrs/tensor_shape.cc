#include <doctest/doctest.h>
#include "op-attrs/tensor_shape.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_broadcast_target_shape(std::unordered_set<TensorShape>)") {
    SUBCASE("target exists in inputs") {
      DataType datatype = DataType::FLOAT;

      TensorShape s1 = TensorShape{
        TensorDims{FFOrdered<size_t>{ 
          1,
        }},
        datatype,
      };

      TensorShape s2 = TensorShape{
        TensorDims{FFOrdered<size_t>{ 
          10, 4, 3
        }},
        datatype,
      };

      TensorShape s3 = TensorShape{
        TensorDims{FFOrdered<size_t>{ 
          4, 1,
        }},
        datatype,
      };

      std::optional<TensorShape> result = get_broadcast_target_shape({s1, s2, s3});
      std::optional<TensorShape> correct = s2;

      CHECK(result == correct);
    }

    SUBCASE("datatypes don't match") {
      TensorDims dims = TensorDims{FFOrdered<size_t>{
        10, 4, 3
      }};

      TensorShape s1 = TensorShape{
        dims,
        DataType::FLOAT,
      };

      TensorShape s2 = TensorShape{
        dims,
        DataType::DOUBLE,
      };

      std::optional<TensorShape> result = get_broadcast_target_shape({s1, s2});
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("inputs is empty") {
      std::optional<TensorShape> result = get_broadcast_target_shape({});
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}

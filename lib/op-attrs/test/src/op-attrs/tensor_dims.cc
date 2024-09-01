#include "op-attrs/tensor_dims.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_dims_is_broadcastable_to(TensorDims, TensorDims)") {

    TensorDims goal = TensorDims{FFOrdered<size_t>{1, 1, 4, 3}};

    SUBCASE("dims match") {
      bool result = tensor_dims_is_broadcastable_to(goal, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr only needs num_dims promotion") {
      TensorDims curr = TensorDims{FFOrdered<size_t>{4, 3}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr only needs dim expansion") {
      TensorDims curr = TensorDims{FFOrdered<size_t>{1, 1, 1, 3}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr needs both num_dims promotion and dim expansion") {
      TensorDims curr = TensorDims{FFOrdered<size_t>{1, 3}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr needs invalid dim promotion") {
      TensorDims curr = TensorDims{FFOrdered<size_t>{1, 1, 2, 3}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("num_dims(goal) < num_dims(curr)") {
      TensorDims curr = TensorDims{FFOrdered<size_t>{1, 1, 10, 4, 3}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = false;

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_broadcast_target_dims(std::unordered_set<TensorDims>)") {
    TensorDims d1 = TensorDims{FFOrdered<size_t>{1, 10, 4, 3}};

    TensorDims d2 = TensorDims{FFOrdered<size_t>{10, 4, 1}};

    SUBCASE("has target in inputs") {
      TensorDims d3 = TensorDims{FFOrdered<size_t>{1, 1, 4, 3}};

      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d2, d3});
      std::optional<TensorDims> correct = d1;

      CHECK(result == correct);
    }

    SUBCASE("has no possible target") {
      TensorDims d3 = TensorDims{FFOrdered<size_t>{1, 1, 1, 4}};

      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d2, d3});
      std::optional<TensorDims> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("has possible target, but not in inputs") {
      TensorDims d3 = TensorDims{FFOrdered<size_t>{1, 1, 1, 4, 3}};

      TensorDims possible_target =
          TensorDims{FFOrdered<size_t>{1, 1, 10, 4, 3}};

      REQUIRE(tensor_dims_is_broadcastable_to(d1, possible_target));
      REQUIRE(tensor_dims_is_broadcastable_to(d2, possible_target));
      REQUIRE(tensor_dims_is_broadcastable_to(d3, possible_target));

      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d2, d3});
      std::optional<TensorDims> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("inputs is empty") {
      std::optional<TensorDims> result = get_broadcast_target_dims({});
      std::optional<TensorDims> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("all inputs are same") {
      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d1, d1, d1, d1});
      std::optional<TensorDims> correct = d1;

      CHECK(result == correct);
    }
  }
}

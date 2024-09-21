#include "utils/sequence.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("seq_head") {
    SUBCASE("seq_head with non-empty sequence") {
      using Seq = seq<1, 2, 3, 4>;
      constexpr int result = seq_head<Seq>::value;
      CHECK(result == 1);
    }

    SUBCASE("seq_head with empty sequence") {
      using Seq = seq<>;
      constexpr int result = seq_head<Seq>::value;
      CHECK(result == -1);
    }
  }

  TEST_CASE("seq_tail") {
    SUBCASE("seq_tail with non-empty sequence") {
      using Seq = seq<1, 2, 3, 4>;
      using ResultType = typename seq_tail<Seq>::type;
      using ExpectedType = seq<2, 3, 4>;
      CHECK(std::is_same<ResultType, ExpectedType>::value);
    }

    SUBCASE("seq_tail with empty sequence") {
      using Seq = seq<>;
      using ResultType = typename seq_tail<Seq>::type;
      using ExpectedType = seq<>;
      CHECK(std::is_same<ResultType, ExpectedType>::value);
    }
  }

  TEST_CASE("seq_prepend") {
    using ResultType = typename FlexFlow::seq_prepend<1, 2, 3>::type;
    using ExpectedType = FlexFlow::seq<1, 2, 3>;
    CHECK(std::is_same<ResultType, ExpectedType>::value);
  }

  TEST_CASE("seq_append") {
    using Seq = seq<1, 2, 3>;
    using ResultType = typename seq_append<Seq, 4>::type;
    using ExpectedType = seq<1, 2, 3, 4>;
    CHECK(std::is_same<ResultType, ExpectedType>::value);
  }

  TEST_CASE("seq_count") {
    using ResultType = seq_count_t<5>;
    using ExpectedType = seq<1, 2, 3, 4, 5>;
    CHECK(!std::is_same<ResultType, ExpectedType>::value);
  }

  TEST_CASE("seq_enumerate_args") {
    using Args = std::tuple<int, float, double>;
    using ResultType = seq_enumerate_args_t<int, float, double>;
    using ExpectedType = seq<0, 1, 2>;
    CHECK(std::is_same<ResultType, ExpectedType>::value);
  }

  // template <int X>
  // int square(std::integral_constant<int, X>) {
  //   return X * X;
  // }

  // TEST_CASE("seq_select") {
  //   SUBCASE("Valid index") {
  //     using Seq = seq<1, 2, 3>;
  //     int result = seq_select(square<int>, 1, seq<1, 2, 3>);
  //     CHECK(result == 4);
  //   }

  //   SUBCASE("Invalid index") {
  //     using Seq = seq<1, 2, 3>;
  //     CHECK_THROWS_AS(seq_select(square<int>, 3, Seq{}), std::runtime_error);
  //   }
  // }

  // TEST_CASE("seq_get") {
  //   SUBCASE("Valid index") {
  //     using Seq = seq<1, 2, 3>;
  //     int result = seq_get(square<int>, 2, Seq{});
  //     CHECK(result == 9);
  //   }

  //   SUBCASE("Invalid index") {
  //     using Seq = seq<1, 2, 3>;
  //     CHECK_THROWS_AS(seq_get(square<int>, 3, Seq{}), std::runtime_error);
  //   }
  // }

  // TEST_CASE("seq_get") {
  //   struct F {
  //     template <int X>
  //     int operator()(std::integral_constant<int, X>) const {
  //       return X * X;
  //     }
  //   };

  //   SUBCASE("Valid index") {
  //     using Seq = seq<1, 2, 3>;
  //     int result = seq_get(F{}, 2, Seq{});
  //     CHECK(result == 9);
  //   }

  //   SUBCASE("Invalid index") {
  //     using Seq = seq<1, 2, 3>;
  //     CHECK_THROWS_AS(seq_get(F{}, 3, Seq{}), std::runtime_error);
  //   }
  // }

  // struct F {
  //   template <int X>
  //   struct type {
  //     using result = std::integral_constant<int, X * X>;
  //   };
  // };

  // TEST_CASE("seq_transform_type") {
  //   using Seq = seq<1, 2, 3>;
  //   using ResultType = seq_transform_type_t<F, Seq>;
  //   using ExpectedType = std::tuple<std::integral_constant<int, 1>,
  //                                   std::integral_constant<int, 4>,
  //                                   std::integral_constant<int, 9>>;
  //   CHECK(std::is_same<ResultType, ExpectedType>::value);
  // }

  // TEST_CASE("seq_transform") {
  //   struct F {
  //     template <int X>
  //     int operator()(std::integral_constant<int, X>) {
  //       return X * X;
  //     }
  //   };

  //   using Seq = seq<1, 2, 3>;
  //   auto result = seq_transform(F{}, Seq{});
  //   std::tuple<int, int, int> expected{1, 4, 9};
  //   CHECK(result == expected);
  // }

  // TEST_CASE("seq_select") {
  //   struct F {
  //     template <int X>
  //     tl::optional<int> operator()(std::integral_constant<int, X>) {
  //       if (X % 2 == 0) {
  //         return X;
  //       } else {
  //         return tl::nullopt;
  //       }
  //     }
  //   };

  //   using Seq = seq<1, 2, 3, 4, 5>;
  //   int result = seq_select(F{}, Seq{});
  //   CHECK(result == 2);
  // }

  // TEST_CASE("seq_get") {
  //   struct F {
  //     template <int X>
  //     int operator()(std::integral_constant<int, X>) {
  //       return X * X;
  //     }
  //   };

  //   using Seq = seq<1, 2, 3, 4, 5>;
  //   int result = seq_get(F{}, 3, Seq{});
  //   CHECK(result == 16);
  // }
}

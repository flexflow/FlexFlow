#include "utils/type_traits_extra/implies.h"
#include "utils/testing.h"
#include <utility>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("implies_v") {
    CHECK(implies_v<std::true_type, std::true_type>);
    CHECK_FALSE(implies_v<std::true_type, std::false_type>);
    CHECK(implies_v<std::false_type, std::true_type>);
    CHECK(implies_v<std::false_type, std::false_type>);
  }

  TEST_CASE_TEMPLATE("implies",
                     T,
                     std::pair<std::true_type, std::true_type>,
                     std::pair<std::true_type, std::false_type>) {
    using LHS = typename T::first_type;
    using RHS = typename T::second_type;
    CHECK(implies_v<LHS, RHS> == implies<LHS, RHS>::value);
  }
}

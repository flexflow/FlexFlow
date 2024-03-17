#include "utils/variant_extra/type/is_in_variant.h"
#include "utils/testing.h"

struct t1 {};
struct t2 {};
struct t3 {};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("is_in_variant_v is true",
                     T,
                     std::pair<t1, std::variant<t1, t2, t3>>,
                     std::pair<t2, std::variant<t1, t2, t3>>,
                     std::pair<t3, std::variant<t1, t2, t3>>,
                     std::pair<t1, std::variant<t1>>,
                     std::pair<t1 const, std::variant<t1 const>>) {
    using L = typename T::first_type;
    using R = typename T::second_type;
    CHECK(is_in_variant_v<L, R>);
    CHECK(is_in_variant_v<L, R const>);
    CHECK(is_in_variant_v<L, R &>);
    CHECK(is_in_variant<L, R>::value);
    CHECK(is_in_variant<L, R const>::value);
    CHECK(is_in_variant<L, R &>::value);
  }

  TEST_CASE_TEMPLATE("is_in_variant_v is false",
                     T,
                     std::pair<t1, std::variant<t2, t3>>,
                     std::pair<t1 &, std::variant<t1>>,
                     std::pair<t1, std::variant<t1 const>>,
                     std::pair<t1 const, std::variant<t1>>) {
    using L = typename T::first_type;
    using R = typename T::second_type;
    CHECK_FALSE(is_in_variant_v<L, R>);
    CHECK_FALSE(is_in_variant<L, R>::value);
  }
}

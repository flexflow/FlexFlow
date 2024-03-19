#include "utils/type_list/functions/contains.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("type_list_contains_v") {
    CHECK(type_list_contains_v<int, type_list<int, float, double>>);
    CHECK(type_list_contains_v<double, type_list<int, float, double>>);
    CHECK(type_list_contains_v<double, type_list<int, float, double> &>);
    CHECK(type_list_contains_v<double, type_list<int, float, double> const>);
    CHECK(type_list_contains_v<int, type_list<int>>);
    CHECK_FALSE(type_list_contains_v<double &, type_list<int, float, double>>);
    CHECK_FALSE(
        type_list_contains_v<double const, type_list<int, float, double>>);
    CHECK_FALSE(type_list_contains_v<long, type_list<int, float, double>>);
    CHECK_FALSE(type_list_contains_v<int, type_list<>>);
  }
}

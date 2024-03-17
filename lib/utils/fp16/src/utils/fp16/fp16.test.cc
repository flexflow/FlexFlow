#include "utils/fp16/fp16.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("half") {
    half h1 = 4.0;
    half h2 = 5.0;
    CHECK(h1 == h1);
    CHECK(h1 != h2);
  }

  TEST_CASE("std::hash<half>") {
    half h1 = 4.0;
    half h2 = 5.0;

    std::hash<half> get_hash{};
    CHECK(get_hash(h1) == get_hash(h1));
    CHECK(get_hash(h1) != get_hash(h2));
  }
}

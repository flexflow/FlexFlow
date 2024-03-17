#include "utils/bidict/algorithms/generate.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("generate_bidict(C const, F const &)") {
    std::unordered_set<int> keys = {1, 2, 3};

    bidict<int, std::string> correct;
    correct.equate(1, std::to_string(1));
    correct.equate(2, std::to_string(2));
    correct.equate(3, std::to_string(3));

    bidict<int, std::string> result =
        generate_bidict(keys, [](int k) { return std::to_string(k); });
    CHECK(result == correct);
  }
}

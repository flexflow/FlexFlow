#include "utils/testing.h"
#include "utils/type_traits_extra/iterator.h"
#include <vector>
#include <unordered_set>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("supports_iterator_tag") {
    CHECK(supports_iterator_tag_v<std::vector<int>, std::random_access_iterator_tag>);
    CHECK_FALSE(supports_iterator_tag_v<std::unordered_set<int>, std::random_access_iterator_tag>);
  }
}

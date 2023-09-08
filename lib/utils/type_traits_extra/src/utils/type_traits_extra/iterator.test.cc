#include "testing.h"
#include "utils/type_traits_extra/iterator.h"
#include <vector>

using namespace FlexFlow {

TEST_CASE("supports_iterator_tag") {
  CHECK(supports_iterator_tag<std::vector<int>, std::random_access_iterator_tag>);
}

}

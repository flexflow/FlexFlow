#include "testing.h"

using namespace FlexFlow;

TEST_CASE("tuple_tail_t") {
  CHECK_SAME_TYPE(tuple_tail_t<1, std::tuple<int>, std::tuple<>);
}

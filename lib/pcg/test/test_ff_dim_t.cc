#include "doctest.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "utils.h"

using namespace FlexFlow;

TEST_CASE("ff_dim_t") {
  CHECK(from_v1(to_v1(ff_dim_t(11))) == 11);
}

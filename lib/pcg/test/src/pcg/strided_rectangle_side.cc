#include "pcg/strided_rectangle_side.h"
#include "pcg/strided_rectangle.h"
#include "test/utils/doctest.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_side_size(StridedRectangleSide)") {
    StridedRectangleSide side{num_points_t{7}, stride_t{5}};

    CHECK(get_side_size(side) == side_size_t{7 * 5});
  }
  TEST_CASE("strided_side_from_size_and_stride") {
    StridedRectangleSide correct{num_points_t{10}, stride_t{3}};
    StridedRectangleSide result =
        strided_side_from_size_and_stride(side_size_t{10 * 3}, stride_t{3});
    CHECK(result == correct);
  }
}

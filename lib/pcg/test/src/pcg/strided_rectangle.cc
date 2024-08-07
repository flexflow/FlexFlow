#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"
#include "test/utils/doctest.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("StridedRectangle - helper functions") {

    StridedRectangleSide s0{num_points_t{7}, stride_t{5}};
    StridedRectangleSide s1{num_points_t{10}, stride_t{2}};
    StridedRectangleSide s2{num_points_t{8}, stride_t{1}};
    StridedRectangle rect{{s0, s1, s2}};

    SUBCASE("get_num_dims") {
      CHECK(get_num_dims(rect) == 3);
    }
    SUBCASE("get_num_points") {
      CHECK(get_num_points(rect) == num_points_t{7 * 8 * 10});
    }

    SUBCASE("get_size") {
      CHECK(get_size(rect) == size_t{(7 * 5) * (10 * 2) * (8 * 1)});
    }
  }
}

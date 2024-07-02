#include "doctest/doctest.h"
#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("StridedRectangleSide") {
    StridedRectangleSide side{num_points_t{7}, 5};

    CHECK(get_side_size(side) == side_size_t{7*5});
    StridedRectangleSide result = strided_side_from_size_and_stride(side_size_t{7*5}, 5);
    CHECK(result == side);
  }

  TEST_CASE("StridedRectangle") {

    StridedRectangleSide s0{num_points_t{7}, 5};
    StridedRectangleSide s1{num_points_t{10}, 2};
    StridedRectangleSide s2{num_points_t{8}, 1};
    
    StridedRectangle rect{{s0, s1, s2}};

    CHECK(get_num_dims(rect) == 3);
    CHECK(get_side_at_idx(rect, ff_dim_t{0}) == s0);
    CHECK(get_side_at_idx(rect, ff_dim_t{1}) == s1);
    CHECK(get_side_at_idx(rect, ff_dim_t{2}) == s2);
  }
}

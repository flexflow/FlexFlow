#include "pcg/strided_rectangle.h"
#include "pcg/strided_rectangle_side.h"
#include "test/utils/doctest.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("StridedRectangle") {
    SUBCASE("constructor sorts the StridedRectangleSides") {
      StridedRectangleSide s0{num_points_t{7}, stride_t{5}};
      StridedRectangleSide s1{num_points_t{10}, stride_t{2}};

      StridedRectangle r0 = StridedRectangle{{s0, s1}};
      StridedRectangle r1 = StridedRectangle{{s1, s0}};
      CHECK(r0 == r1);
      CHECK(r1.get_sides() == std::vector<StridedRectangleSide>{s0, s1});
      CHECK(r1.get_sides() != std::vector<StridedRectangleSide>{s1, s0});
    }

    SUBCASE("helper functions") {
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
    }
  }
}

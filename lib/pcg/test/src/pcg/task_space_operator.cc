#include "pcg/task_space_operator.h"
#include "utils/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("task_space_operator functions") {
    SUBCASE("get_fragment_coordinates") {

      SUBCASE("2D Task") {

        TaskSpaceOperator task =
            TaskSpaceOperator{{num_points_t{2}, num_points_t{2}}};

        std::unordered_set<TaskSpaceCoordinate> correct = {
            {TaskSpaceCoordinate{{0, 0}},
             TaskSpaceCoordinate{{0, 1}},
             TaskSpaceCoordinate{{1, 0}},
             TaskSpaceCoordinate{{1, 1}}}};
        std::unordered_set<TaskSpaceCoordinate> result =
            get_fragment_coordinates(task);
        CHECK(correct == result);
      }
      SUBCASE("3D Task") {

        TaskSpaceOperator task = TaskSpaceOperator{
            {num_points_t{1}, num_points_t{2}, num_points_t{2}}};

        std::unordered_set<TaskSpaceCoordinate> correct = {
            {TaskSpaceCoordinate{{0, 0, 0}},
             TaskSpaceCoordinate{{0, 0, 1}},
             TaskSpaceCoordinate{{0, 1, 0}},
             TaskSpaceCoordinate{{0, 1, 1}}}};
        std::unordered_set<TaskSpaceCoordinate> result =
            get_fragment_coordinates(task);
        CHECK(correct == result);
      }
    }
    SUBCASE("get_maximum_fragment_coordinate") {

      SUBCASE("2D Task") {

        TaskSpaceOperator task =
            TaskSpaceOperator{{num_points_t{3}, num_points_t{2}}};

        TaskSpaceCoordinate correct = TaskSpaceCoordinate{{2, 1}};
        TaskSpaceCoordinate result = get_maximum_fragment_coordinate(task);
        CHECK(correct == result);
      }
      SUBCASE("3D Task") {

        TaskSpaceOperator task = TaskSpaceOperator{
            {num_points_t{3}, num_points_t{2}, num_points_t{4}}};

        TaskSpaceCoordinate correct = TaskSpaceCoordinate{{2, 1, 3}};
        TaskSpaceCoordinate result = get_maximum_fragment_coordinate(task);
        CHECK(correct == result);
      }
    }
  }
}

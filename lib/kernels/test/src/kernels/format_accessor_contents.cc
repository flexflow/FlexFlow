#include "kernels/format_accessor_contents.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/local_cpu_allocator.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("format_accessor_r_contents(GenericTensorAccessorR)") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("accessor is 1d") {
      GenericTensorAccessorR accessor =
          create_1d_accessor_r_with_contents<int32_t>({1, 2, 3, 2},
                                                      cpu_allocator);

      std::string correct = "[1 2 3 2]";

      std::string result = format_accessor_r_contents(accessor);

      CHECK(result == correct);
    }

    SUBCASE("accessor is 2d") {
      GenericTensorAccessorR accessor =
          create_2d_accessor_r_with_contents<int32_t>(
              {
                  {1, 2, 3, 5},
                  {4, 3, 3, 2},
                  {1, 1, 5, 8},
              },
              cpu_allocator);

      std::string correct = "[\n"
                            "  [1 2 3 5]\n"
                            "  [4 3 3 2]\n"
                            "  [1 1 5 8]\n"
                            "]";

      std::string result = format_accessor_r_contents(accessor);

      CHECK(result == correct);
    }

    SUBCASE("accessor is 3d") {
      GenericTensorAccessorR accessor =
          create_3d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {1, 2, 3, 6},
                      {4, 3, 3, 9},
                      {1, 1, 5, 1},
                  },
                  {
                      {4, 1, 8, 7},
                      {9, 4, 2, 4},
                      {1, 0, 0, 6},
                  },
                  {
                      {2, 1, 1, 9},
                      {1, 3, 6, 2},
                      {1, 9, 8, 9},
                  },
              },
              cpu_allocator);

      std::string correct = "[\n"
                            "  [\n"
                            "    [1 2 3 6]\n"
                            "    [4 3 3 9]\n"
                            "    [1 1 5 1]\n"
                            "  ]\n"
                            "  [\n"
                            "    [4 1 8 7]\n"
                            "    [9 4 2 4]\n"
                            "    [1 0 0 6]\n"
                            "  ]\n"
                            "  [\n"
                            "    [2 1 1 9]\n"
                            "    [1 3 6 2]\n"
                            "    [1 9 8 9]\n"
                            "  ]\n"
                            "]";

      std::string result = format_accessor_r_contents(accessor);

      CHECK(result == correct);
    }

    SUBCASE("accessor is 4d") {
      GenericTensorAccessorR accessor =
          create_4d_accessor_r_with_contents<int32_t>(
              {
                  {
                      {
                          {2, 1, 1, 9},
                          {1, 3, 6, 2},
                          {1, 9, 8, 9},
                      },
                      {
                          {9, 2, 7, 6},
                          {7, 2, 1, 1},
                          {2, 8, 5, 6},
                      },
                  },
                  {
                      {
                          {1, 2, 3, 6},
                          {4, 3, 3, 9},
                          {1, 1, 5, 1},
                      },
                      {
                          {4, 1, 8, 7},
                          {9, 4, 2, 4},
                          {1, 0, 0, 6},
                      },
                  },
              },
              cpu_allocator);

      std::string correct = "[\n"
                            "  [\n"
                            "    [\n"
                            "      [2 1 1 9]\n"
                            "      [1 3 6 2]\n"
                            "      [1 9 8 9]\n"
                            "    ]\n"
                            "    [\n"
                            "      [9 2 7 6]\n"
                            "      [7 2 1 1]\n"
                            "      [2 8 5 6]\n"
                            "    ]\n"
                            "  ]\n"
                            "  [\n"
                            "    [\n"
                            "      [1 2 3 6]\n"
                            "      [4 3 3 9]\n"
                            "      [1 1 5 1]\n"
                            "    ]\n"
                            "    [\n"
                            "      [4 1 8 7]\n"
                            "      [9 4 2 4]\n"
                            "      [1 0 0 6]\n"
                            "    ]\n"
                            "  ]\n"
                            "]";

      std::string result = format_accessor_r_contents(accessor);

      CHECK(result == correct);
    }
  }
}

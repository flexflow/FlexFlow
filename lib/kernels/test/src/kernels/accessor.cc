#include "kernels/accessor.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/local_cpu_allocator.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("calculate_accessor_offset") {
    SUBCASE("one dimension") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{4_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              13_p,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 4_n;

      CHECK(result == correct);
    }

    SUBCASE("2d tensor is row-major") {
      positive_int num_rows = 5_p;
      positive_int num_cols = 6_p;

      TensorDims shape = TensorDims{
          FFOrdered{
              num_rows,
              num_cols,
          },
      };

      CHECK(calculate_accessor_offset(TensorDimsCoord{FFOrdered{0_n, 0_n}},
                                      shape) == 0_n);
      CHECK(calculate_accessor_offset(TensorDimsCoord{FFOrdered{1_n, 0_n}},
                                      shape) == num_cols);
      CHECK(calculate_accessor_offset(TensorDimsCoord{FFOrdered{0_n, 1_n}},
                                      shape) == 1_p);
    }

    SUBCASE("multiple dimensions") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{2_n, 4_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              5_p,
              6_p,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 2_n * 6_n + 4_n;

      CHECK(result == correct);
    }

    SUBCASE("zero dimensions") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered<nonnegative_int>{}};
      TensorDims shape = TensorDims{FFOrdered<positive_int>{}};

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("index and shape dimensions do not match") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{1_n, 2_n, 4_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              5_p,
              6_p,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }

    SUBCASE("out of bounds index") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{2_n, 6_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              5_p,
              6_p,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }
  }

  TEST_CASE("accessor_get_only_value") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("returns the value if the accessor only contains one value") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
              },
          },
          cpu_allocator);

      float result = accessor_get_only_value<DataType::FLOAT>(input);
      float correct = 12;

      CHECK(result == correct);
    }

    SUBCASE("throws an error if the requested type does not match the type in "
            "the accessor") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
              },
          },
          cpu_allocator);

      CHECK_THROWS(accessor_get_only_value<DataType::DOUBLE>(input));
    }

    SUBCASE("throws an error if the accessor contains multiple values") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
                  {12},
              },
          },
          cpu_allocator);

      CHECK_THROWS(accessor_get_only_value<DataType::FLOAT>(input));
    }
  }
}

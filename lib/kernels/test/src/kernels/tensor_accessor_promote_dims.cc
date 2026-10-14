#include "kernels/tensor_accessor_promote_dims.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_accessor_promote_dims") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("dims are promotable") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {{
                   3,
               },
               {
                   3,
               },
               {
                   6,
               }},
              {{
                   0,
               },
               {
                   -1,
               },
               {
                   0.75,
               }},
          },
          cpu_allocator);

      TensorDims to = TensorDims{
          FFOrdered<positive_int>{
              2_p,
              1_p,
              1_p,
              3_p,
          },
      };

      GenericTensorAccessorR result = tensor_accessor_promote_dims(input, to);

      GenericTensorAccessorR correct =
          create_4d_accessor_r_with_contents<float>(
              {
                  {
                      {
                          {
                              {3, 3, 6},
                          },
                      },
                  },
                  {
                      {
                          {
                              {0, -1, 0.75},
                          },
                      },
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_r_contents(result)));
    }

    SUBCASE("dims are not promotable") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {{
                   3,
               },
               {
                   3,
               },
               {
                   6,
               }},
              {{
                   0,
               },
               {
                   -1,
               },
               {
                   0.75,
               }},
          },
          cpu_allocator);

      TensorDims to = TensorDims{
          FFOrdered<positive_int>{
              6_p,
              1_p,
              1_p,
              1_p,
          },
      };

      CHECK_THROWS(tensor_accessor_promote_dims(input, to));
    }
  }
}

#include "op-attrs/tensor_shape.h"
#include "test/op-attrs/doctest.h"

using namespace FlexFlow;

TEST_CASE("TensorShape Tests") {
  SUBCASE("Construction") {
    TensorDims dims = {1, 2, 3, 4};
    TensorShape tensor_shape(dims, DataType::FLOAT);

    CHECK(tensor_shape.dims == dims);
    CHECK(tensor_shape.data_type == DataType::FLOAT);
  }

  SUBCASE("Accessors") {
    TensorDims dims = {5, 10, 15};
    TensorShape tensor_shape(dims, DataType::INT64);

    CHECK(tensor_shape.at(ff_dim_t(0)) == 5);
    CHECK(tensor_shape.at(ff_dim_t(1)) == 10);
    CHECK(tensor_shape.at(ff_dim_t(2)) == 15);

    // Alternatively, you can use the [] operator for the same purpose
    CHECK(tensor_shape[ff_dim_t(0)] == 5);
    CHECK(tensor_shape[ff_dim_t(1)] == 10);
    CHECK(tensor_shape[ff_dim_t(2)] == 15);
  }
}

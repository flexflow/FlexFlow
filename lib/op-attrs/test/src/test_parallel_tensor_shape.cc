

#include "op-attrs/parallel_tensor_shape.h"

#include "test/op-attrs/doctest.h"

using namespace FlexFlow;

TEST_CASE("ParallelTensorShape Tests") {
  SUBCASE("Construction") {
    ParallelTensorDims dims{{4, 8}};
    DataType data_type = DataType::FLOAT;
    ParallelTensorShape tensor_shape(dims, data_type);

    //    CHECK(tensor_shape.dims == dims);
    CHECK(tensor_shape.data_type == data_type);
  }

  SUBCASE("Accessors") {
    ParallelTensorDims dims{{2, 4}};
    DataType data_type = DataType::INT64;
    ParallelTensorShape tensor_shape(dims, data_type);

    CHECK(tensor_shape.num_dims() == 3);
    CHECK(tensor_shape.at(ff_dim_t(0)).size == 2);
    CHECK(tensor_shape.at(ff_dim_t(0)).degree == 1);
    CHECK(tensor_shape.at(ff_dim_t(1)).size == 4);
    CHECK(tensor_shape.at(ff_dim_t(1)).degree == 1);

    // Alternatively, you can use the [] operator for the same purpose
    CHECK(tensor_shape[ff_dim_t(0)].size == 2);
    CHECK(tensor_shape[ff_dim_t(0)].degree == 1);
    CHECK(tensor_shape[ff_dim_t(1)].size == 4);
    CHECK(tensor_shape[ff_dim_t(1)].degree == 1);
  }

  SUBCASE("Helper Functions") {
    ParallelTensorDims dims{{2, 4}};
    DataType data_type = DataType::INT64;
    ParallelTensorShape tensor_shape(dims, data_type);

    CHECK(get_num_replica_dims(tensor_shape) == 1);
    CHECK(get_num_replicas(tensor_shape) == 1);

    std::vector<TensorShape> tensor_shapes = get_tensor_shapes_unsafe(
        std::vector<ParallelTensorShape>{tensor_shape});
    CHECK(tensor_shapes.size() == 1);
    CHECK(tensor_shapes[0].dims.size() == 3);
    CHECK(tensor_shapes[0].data_type == data_type);
  }
}

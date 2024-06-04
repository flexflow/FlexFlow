#include "test/utils/doctest.h"
#include "pcg/parallel_computation_graph_builder.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ParallelComputationGraphBuilder") {
    ParallelComputationGraphBuilder b;

    size_t batch_size = 2;

    TensorShape unpar_input_shape = {
        TensorDims{FFOrdered<size_t>{batch_size, 3, 10, 10}},
        DataType::FLOAT,
    };
  
    ParallelTensorShape input_shape = lift_to_parallel_with_degrees(unpar_input_shape, SumDegree{1}, DiscardCopyDegree{1}, FFOrdered<int>{2, 1, 1, 1});

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);

    parallel_tensor_guid_t output = b.conv2d(input, 
             /*outChannels=*/5,
             /*kernelH=*/3,
             /*kernelW=*/3,
             /*strideH=*/1,
             /*strideW=*/1,
             /*paddingH=*/0,
             /*paddingW=*/0);

    CHECK(get_parallel_layers(b.pcg).size() == 1);
  };
}

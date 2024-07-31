#include "doctest/doctest.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ComputationGraphBuilder") {
    ComputationGraphBuilder b;

    size_t batch_size = 2;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{batch_size, 3, 10, 10}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_tensor(input_shape, CreateGrad::YES);
    tensor_guid_t output = b.conv2d(input,
                                    /*outChannels=*/5,
                                    /*kernelH=*/3,
                                    /*kernelW=*/3,
                                    /*strideH=*/1,
                                    /*strideW=*/1,
                                    /*paddingH=*/0,
                                    /*paddingW=*/0);
    // ComputationGraph cg = b.computation_graph;
    // CHECK(get_layers(cg).size() == 1);
  }
}

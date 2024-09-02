#include <doctest/doctest.h>
#include "pcg/file_format/v1/v1_computation_graph.h"
#include "pcg/computation_graph_builder.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("V1ComputationGraph") {
    ComputationGraph cg = [] {
      ComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{
          12, 16,
        }},
        DataType::FLOAT,
      };

      tensor_guid_t input = b.create_tensor(input_shape, CreateGrad::YES);
      tensor_guid_t mm_output = b.dense(input, 8);
      tensor_guid_t relu_output = b.relu(mm_output);

      return b.computation_graph;
    }();

    V1ComputationGraph v1_cg = to_v1(cg);
    nlohmann::json j = v1_cg;
  }
}

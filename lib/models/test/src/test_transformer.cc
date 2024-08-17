#include "doctest/doctest.h"
#include "models/transformer.h"
#include "pcg/computation_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_transformer_computation_graph") {
    TransformerConfig config =
        TransformerConfig{1024, 1024, 16, 12, 512, 64, 1e-05};

    ComputationGraph result = get_transformer_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 88;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}

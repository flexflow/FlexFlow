#include "models/inception_v3/inception_v3.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_inception_v3_computation_graph") {
    InceptionV3Config config = get_default_inception_v3_config();

    ComputationGraph result = get_inception_v3_computation_graph(config);

    SUBCASE("num layers") {
      // int result_num_layers = get_layers(result).size();
      int correct_num_layers = -1;
      // CHECK(result_num_layers == correct_num_layers);
    }
  }
}

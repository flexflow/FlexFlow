#include "models/dlrm.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_dlrm_computation_graph") {
    DLRMConfig config = get_default_dlrm_config();

    ComputationGraph result = get_dlrm_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 0;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}

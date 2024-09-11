#include "models/candle_uno/candle_uno.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_candle_uno_computation_graph") {
    CandleUnoConfig config = get_default_candle_uno_config();

    ComputationGraph result = get_candle_uno_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 98;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}

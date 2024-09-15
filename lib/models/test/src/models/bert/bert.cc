#include "models/bert/bert.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_bert_computation_graph") {
    BertConfig config = get_default_bert_config();

    ComputationGraph result = get_bert_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 269;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}

#include "models/bert/bert.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_bert_computation_graph") {

    SUBCASE("default config") {
      BertConfig config = get_default_bert_config();

      ComputationGraph result = get_bert_computation_graph(config);

      SUBCASE("num layers") {
        int result_num_layers = get_layers(result).size();
        int correct_num_layers = 245;
        CHECK(result_num_layers == correct_num_layers);
      }
    }

    SUBCASE("throws on position_embedding_type != absolute as other values are currently unsupported") {
      BertConfig config = [] {
        BertConfig c = get_default_bert_config();
        c.position_embedding_type = "relative_key";
        return c;
      }();

      CHECK_THROWS(get_bert_computation_graph(config));  
    }
  }
}

#include "models/transformer.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_transformer_computation_graph") {
    TransformerConfig config = TransformerConfig{/*num_features=*/512,
                                                 /*sequence_length=*/512,
                                                 /*batch_size=*/64,
                                                 /*dim_feedforward=*/2048,
                                                 /*num_heads=*/8,
                                                 /*num_encoder_layers=*/6,
                                                 /*num_decoder_layers=*/6,
                                                 /*dropout=*/0.1,
                                                 /*layer_norm_eps=*/1e-05,
                                                 /*vocab_size=*/64};

    ComputationGraph result = get_transformer_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 317;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}

#include "doctest/doctest.h"
#include "models/transformer.h"
#include "pcg/computation_graph.h"

using namespace ::FlexFlow;

/*
  TransformerConfig(size_t const &num_features,
                    size_t const &sequence_length,
                    size_t const &batch_size,
                    size_t const &dim_feedforward,
                    size_t const &num_heads,
                    size_t const &num_encoder_layers,
                    size_t const &num_decoder_layers,
                    float const &dropout,
                    float const &layer_norm_eps,
                    size_t const &vocab_size)
*/

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_transformer_computation_graph") {
    TransformerConfig config =
        TransformerConfig{512, 512, 64, 2048, 8, 6, 6, 0.1, 1e-05, 64};

    ComputationGraph result = get_transformer_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 166;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}

#include "doctest/doctest.h"
#include "models/transformer.h"
#include "pcg/computation_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("TransformerTest") {
    TransformerConfig config(1024, 1024, 16, 12, 512, 64);
    ComputationGraph cg = get_transformer_computation_graph(config);
    CHECK(get_layers(cg).size() == 33);
  }
}

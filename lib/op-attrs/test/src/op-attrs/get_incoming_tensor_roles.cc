#include "op-attrs/get_incoming_tensor_roles.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE(
      "get_incoming_tensor_roles(ComputationGraphOpAttrs, int num_incoming)") {
    SUBCASE("Concat") {
      int num_incoming = 4;
      ComputationGraphOpAttrs attrs =
          ComputationGraphOpAttrs{ConcatAttrs{ff_dim_t{0}, num_incoming}};

      std::vector<IncomingTensorRole> result =
          get_incoming_tensor_roles(attrs, num_incoming);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
          IncomingTensorRole::INPUT,
          IncomingTensorRole::INPUT,
          IncomingTensorRole::INPUT,
      };

      CHECK(result == correct);
    }
  }
}

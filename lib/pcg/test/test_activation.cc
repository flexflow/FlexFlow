#include "doctest.h"
#include "pcg/file_format/v1/activation.h"
#include "utils.h"

using namespace FlexFlow;

TEST_CASE("Activation") {
  V1Activation v10 = to_v1(Activation::RELU);
  CHECK(from_v1(v10) == Activation::RELU);
  CHECK(str(json(v10)) == "\"RELU\"");

  V1Activation v11 = to_v1(Activation::SIGMOID);
  CHECK(from_v1(v11) == Activation::SIGMOID);
  CHECK(str(json(v11)) == "\"SIGMOID\"");

  V1Activation v12 = to_v1(Activation::TANH);
  CHECK(from_v1(v12) == Activation::TANH);
  CHECK(str(json(v12)) == "\"TANH\"");

  V1Activation v13 = to_v1(Activation::GELU);
  CHECK(from_v1(v13) == Activation::GELU);
  CHECK(str(json(v13)) == "\"GELU\"");
}

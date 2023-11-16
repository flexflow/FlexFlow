#include "doctest.h"
#include "pcg/file_format/v1/create_grad.h"
#include "utils.h"

using namespace FlexFlow;

TEST_CASE("CreateGrad") {
  V1CreateGrad v10 = to_v1(CreateGrad::YES);
  CHECK(from_v1(v10) == CreateGrad::YES);
  CHECK(str(json(v10)) == "\"YES\"");

  V1CreateGrad v11 = to_v1(CreateGrad::NO);
  CHECK(from_v1(v11) == CreateGrad::NO);
  CHECK(str(json(v11)) == "\"NO\"");
}

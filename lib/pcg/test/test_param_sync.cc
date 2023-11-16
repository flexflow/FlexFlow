#include "doctest.h"
#include "pcg/file_format/v1/param_sync.h"
#include "utils.h"

using namespace FlexFlow;

TEST_CASE("ParamSync") {
  V1ParamSync v10 = to_v1(ParamSync::PS);
  CHECK(from_v1(v10) == ParamSync::PS);
  CHECK(str(json(v10)) == "\"PARAM_SERVER\"");

  V1ParamSync v11 = to_v1(ParamSync::NCCL);
  CHECK(from_v1(v11) == ParamSync::NCCL);
  CHECK(str(json(v11)) == "\"NCCL\"");
}

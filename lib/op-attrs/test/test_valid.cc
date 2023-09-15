#include "doctest/doctest.h"
#include "op-attrs/ops/gather.h"
#include <rapidcheck.h>
#include <vector>

using namespace FlexFlow;

TEST_CASE("GatherAttrs::is_valid") {

  GatherAttrs g{ff_dim_t(2)};

  TensorDims tds({2, 2});
  ParallelTensorShape p(tds, DataType::FLOAT);
  CHECK(g.is_valid(p, p));
};

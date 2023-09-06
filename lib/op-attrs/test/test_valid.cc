#include "op-attrs/ops/gather.h"
#include "doctest/doctest.h"
#include <rapidcheck.h>
#include <vector>

using namespace FlexFlow;

TEST_CASE("gather_valid") {

  GatherAttrs g{ff_dim_t(2)}; 

  TensorDims tds({2,2}); 
  ParallelTensorShape p(tds, DataType::FLOAT); 
  RC_ASSERT(g.is_valid(p, p));  

};


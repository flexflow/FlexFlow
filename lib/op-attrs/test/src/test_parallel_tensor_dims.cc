#include "op-attrs/parallel_tensor_dims.h"
#include "test/op-attrs/doctest.h"

using namespace FlexFlow;

TEST_CASE("ParallelTensorDims Tests") {
  FFOrdered<size_t> dims1 = {1, 2, 3, 4};
  ParallelTensorDims parallel_tensor_dims{dims1};
  SUBCASE("Size and Accessors ") {
    CHECK(parallel_tensor_dims.num_dims() == dims1.size());
    CHECK(parallel_tensor_dims.at(ff_dim_t(1)).size == dims1[ff_dim_t(1)]);
  }

  SUBCASE("Iterators begin") {
    FFOrdered<ParallelDim>::iterator it = parallel_tensor_dims.begin();
    CHECK((*it).size == 1);
    ++it;
    CHECK((*it).size == 2);

    FFOrdered<ParallelDim>::const_iterator cit =
    parallel_tensor_dims.cbegin(); CHECK((*cit).size == 1);
    ++cit;
    CHECK((*cit).size == 2);

    FFOrdered<ParallelDim>::reverse_iterator rit =
        parallel_tensor_dims.rbegin();
    CHECK((*rit).size == 4);
    ++rit;
    CHECK((*rit).size == 3);

    FFOrdered<ParallelDim>::const_reverse_iterator crit =
        parallel_tensor_dims.crbegin();

    CHECK((*crit).size == 4);
    ++rit;
    CHECK((*crit).size == 3);
  }
}

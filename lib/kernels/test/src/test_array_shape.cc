#include "doctest.h"
#include "kernels/array_shape.h" // Assuming this is where your ArrayShape is
#include "kernels/legion_dim.h"

using namespace FlexFlow;

TEST_CASE("ArrayShape Initialization and Basic Functions") {
  std::vector<std::size_t> dims = {2, 3, 4};
  ArrayShape shape(dims);
  CHECK(shape.get_dim() == 3);
  CHECK(shape.get_volume() == 24);
  CHECK(shape.num_elements() == 24);
  CHECK(shape.num_dims() == 3);
  CHECK(shape[legion_dim_t(1)] == 3);
  CHECK(shape.at(legion_dim_t(2)) == 4);
  CHECK(shape.at(ff_dim_t(2)) == 4);
}

TEST_CASE("Negative Indices and Optional Indexing") {
  std::vector<std::size_t> dims = {2, 3, 4};
  ArrayShape shape(dims);

  CHECK(shape.neg_idx(-1) == legion_dim_t(2));
  CHECK(shape.neg_idx(-3) == legion_dim_t(0));

  CHECK(shape.at_maybe(0) == 2);
  CHECK(shape.at_maybe(2) == 4);
  CHECK(!shape.at_maybe(5).has_value());
}

TEST_CASE("Reversed Dim Order and Sub-shape") {
  using namespace FlexFlow;

  std::vector<std::size_t> dims = {2, 3, 4};
  ArrayShape shape(dims);

  ArrayShape reversed = shape.reversed_dim_order();
  CHECK(reversed[legion_dim_t(0)] == 4);
  CHECK(reversed[legion_dim_t(1)] == 3);
  CHECK(reversed[legion_dim_t(2)] == 2);

  ArrayShape sub = shape.sub_shape(legion_dim_t(0), legion_dim_t(2));
  CHECK(sub.get_dim() == 2);
  CHECK(sub[legion_dim_t(0)] == 2);
  CHECK(sub[legion_dim_t(1)] == 3);
}

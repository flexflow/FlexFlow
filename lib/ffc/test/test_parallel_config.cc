#include "flexflow/config.h"
#include "flexflow/model.h"
#include "gtest/gtest.h"

using namespace FlexFlow;

TEST(change_data_parallel_dimensionality, basic_reduce) {
  ParallelConfig pc = get_basic_data_parallel_config(8, 4);

  ParallelConfig expected = get_basic_data_parallel_config(8, 2);

  ParallelConfig result = pc.change_data_parallel_dimensionality(2);

  EXPECT_EQ(result, expected);
}

TEST(change_data_parallel_dimensionality, basic_expand) {
  ParallelConfig pc = get_basic_data_parallel_config(8, 2);

  ParallelConfig expected = get_basic_data_parallel_config(8, 4);

  ParallelConfig result = pc.change_data_parallel_dimensionality(4);

  EXPECT_EQ(result, expected);
}

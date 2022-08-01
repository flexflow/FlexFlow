#include "flexflow/utils/random_utils.h"
#include "gtest/gtest.h"

TEST(select_random, basic) {
  std::vector<int> values{1, 2, 3, 4};
  std::vector<float> weights{0.1, 0.2, 0.3, 0.4};

  EXPECT_EQ(select_random_determistic(values, weights, 0.05), 1);
  EXPECT_EQ(select_random_determistic(values, weights, 0.25), 2);
  EXPECT_EQ(select_random_determistic(values, weights, 0.5), 3);
  EXPECT_EQ(select_random_determistic(values, weights, 0.9), 4);
}

TEST(select_random, bounds) {
  std::vector<int> values{1, 2, 3};
  std::vector<float> weights{0.2, 0.3, 0.5};

  EXPECT_EQ(select_random_determistic(values, weights, 0.0), 1);
  EXPECT_EQ(select_random_determistic(values, weights, 0.2), 2);
  EXPECT_EQ(select_random_determistic(values, weights, 0.5), 3);
  EXPECT_EQ(select_random_determistic(values, weights, 1.0), 3);
}

TEST(select_random, singleton) {
  std::vector<int> values{1};
  std::vector<float> weights{1.0};

  EXPECT_EQ(select_random_determistic(values, weights, 0.0), 1);
  EXPECT_EQ(select_random_determistic(values, weights, 0.5), 1);
  EXPECT_EQ(select_random_determistic(values, weights, 1.0), 1);
}

TEST(select_random, empty) {
  std::vector<int> values{};
  std::vector<float> weights{};
  EXPECT_THROW(select_random_determistic(values, weights, 0.5),
               std::invalid_argument);
}

TEST(select_random, unnormalized_weights) {
  std::vector<int> values{1, 2, 3};
  std::vector<float> weights{1.0, 2.0, 2.0};

  EXPECT_EQ(select_random_determistic(values, weights, 0.1), 1);
  EXPECT_EQ(select_random_determistic(values, weights, 0.5), 2);
  EXPECT_EQ(select_random_determistic(values, weights, 0.9), 3);
}

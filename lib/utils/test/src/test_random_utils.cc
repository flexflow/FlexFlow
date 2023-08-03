
#include "doctest.h"
#include "utils/random_utils.h"

TEST_CASE("select_random") {
  std::vector<int> values = {1, 2, 3, 4, 5};

  SUBCASE("Select random value") {
    int result = select_random(values);

    CHECK(std::find(values.begin(), values.end(), result) != values.end());
  }

  SUBCASE("Invalid arguments") {
    std::vector<float> weights = {0.1f, 0.3f, 0.2f};
    std::cout << select_random(values, weights);
    CHECK(select_random(values, weights) == 3);
  }
}

TEST_CASE("select_random - Weighted Random Selection") {
  SUBCASE("Test with equal weights") {
    std::vector<int> values = {1, 2, 3, 4, 5};
    std::vector<float> weights = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<int> counts(values.size(), 0);
    int const numIterations = 10000;
    for (int i = 0; i < numIterations; i++) {
      int selected = select_random(values, weights);
      counts[selected - 1]++;
    }

    for (int i = 0; i < values.size(); i++) {
      float expectedProbability = 1.0f / values.size();
      float observedProbability = static_cast<float>(counts[i]) / numIterations;
      CHECK(observedProbability ==
            doctest::Approx(expectedProbability).epsilon(0.01f));
    }
  }

  SUBCASE("Test with different weights") {
    std::vector<int> values = {1, 2, 3, 4, 5};
    std::vector<float> weights = {0.1f, 0.2f, 0.3f, 0.2f, 0.2f};

    std::vector<int> counts(values.size(), 0);
    int const numIterations = 10000;
    for (int i = 0; i < numIterations; i++) {
      int selected = select_random(values, weights);
      counts[selected - 1]++;
    }

    float totalWeight = 0.0f;
    for (float weight : weights) {
      totalWeight += weight;
    }

    for (int i = 0; i < values.size(); i++) {
      float expectedProbability = weights[i] / totalWeight;
      float observedProbability = static_cast<float>(counts[i]) / numIterations;
      CHECK(observedProbability ==
            doctest::Approx(expectedProbability).epsilon(0.01f));
    }
  }
}

#include "utils/random_utils.h"
#include "test/utils/doctest.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/containers/repeat.h"
#include "utils/containers/sum.h"
#include "utils/containers/zip.h"
#include <algorithm>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("select_random(std::vector<T>)") {
    std::vector<int> values = {1, 2, 3, 4, 5};

    SUBCASE("selected value is in container") {
      SUBCASE("equal weights") {
        int result = select_random(values);
        CHECK(contains(values, result));
      }

      SUBCASE("unequal weights") {
        std::vector<float> weights = {0.1f, 0.3f, 0.2f, 0.2f, 0.2f};
        int result = select_random(values, weights);
        CHECK(contains(values, result));
      }
    }

    SUBCASE("correct distribution") {
      auto check_probabilities = [](std::vector<int> const &values,
                                    std::vector<float> const &weights) {
        int num_iterations = 10'000;
        std::vector<int> trials = repeat(
            num_iterations, [&]() { return select_random(values, weights); });

        for (auto const [v, w] : zip(values, weights)) {
          float expectedProbability = w / sum(weights);
          int num_occurrences =
              filter(trials, [&](int c) { return (c == v); }).size();
          float observedProbability =
              static_cast<float>(num_occurrences) / num_iterations;
          CHECK(observedProbability ==
                doctest::Approx(expectedProbability).epsilon(0.01f));
        }
      };

      SUBCASE("equal weights") {
        std::vector<float> weights = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        check_probabilities(values, weights);
      }

      SUBCASE("unequal weights") {
        std::vector<float> weights = {0.1f, 0.2f, 0.3f, 0.2f, 0.2f};
        check_probabilities(values, weights);
      }
    }
  }
}

#include "benchmark/substitutions/apply_substitution/apply_substitution.h"
#include "benchmark/substitutions/pcg_pattern.h"
#include "utils/benchmark_utils/benchmark_main.h"
#include <random>

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  std::map<std::string, std::function<void(bool)>> benchmarks = {
      {
          "find_pattern_matches_small",
          benchmark_find_pattern_matches_small,
      },
      {
          "find_pattern_matches_medium",
          benchmark_find_pattern_matches_medium,
      },
      /*
       * currently times out
      {
          "find_pattern_matches_large",
          benchmark_find_pattern_matches_large,
      },
      */
      {
          "apply_substitution",
          benchmark_apply_substitution,
      },
  };

  benchmark_main(argc, argv, benchmarks);
}

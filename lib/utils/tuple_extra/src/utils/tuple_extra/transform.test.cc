#include "utils/tuple_extra/transform.h"
#include "utils/overload/overload.h"
#include "utils/testing.h"
#include <sstream>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform(std::tuple<Ts...> const &, F const &f)") {
    auto example_function = overload{
        [](std::string const &s) -> int { return s.size(); },
        [](int i) -> std::string {
          std::ostringstream oss;
          oss << i;
          return oss.str();
        },
    };

    std::tuple<int, std::string, std::string> input = {1, "a", "aa"};
    std::tuple<std::string, int, int> correct = std::tuple{
        example_function(std::get<0>(input)),
        example_function(std::get<1>(input)),
        example_function(std::get<2>(input)),
    };
    std::tuple<std::string, int, int> result =
        transform(input, example_function);

    CHECK_EQ(result, correct);
  }
}

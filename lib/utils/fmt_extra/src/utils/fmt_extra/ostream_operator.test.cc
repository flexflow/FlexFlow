#include "utils/fmt_extra/ostream_operator.h"
#include "utils/testing.h"
#include <sstream>
#include <string>

struct my_example_type {};

std::string format_as(my_example_type const &) {
  return "my_example_type{}";
}

struct my_example_undelegated_type {};

std::string format_as(my_example_undelegated_type const &) {
  return "used fmt";
}

std::ostream &operator<<(std::ostream &s, my_example_undelegated_type const &) {
  return (s << "used ostream operator");
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ostream delegation for custom type") {
    my_example_type x;

    std::ostringstream oss;
    oss << x;
    std::string result = oss.str();
    std::string correct = fmt::to_string(x);

    CHECK(result == correct);
  }

  TEST_CASE("explicitly preventing ostream delegation for custom type") {
    my_example_undelegated_type x;

    std::ostringstream oss;
    oss << x;
    std::string result = oss.str();
    std::string incorrect = fmt::to_string(x);

    CHECK(result != incorrect);
  }
}

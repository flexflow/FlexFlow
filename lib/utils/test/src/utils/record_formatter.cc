#include "utils/record_formatter.h"
#include <doctest/doctest.h>

std::string formatRecord(RecordFormatter const &formatter) {
  std::ostringstream oss;
  oss << formatter;
  return oss.str();
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RecordFormatter") {
    RecordFormatter formatter;
    SUBCASE("Appending string") {
      formatter << "Hello";
      formatter << "World";
      CHECK(formatRecord(formatter) == "{ Hello | World }");
    }

    SUBCASE("Appending integer and float") {
      formatter << 42;
      formatter << 3.14f;
      CHECK(formatRecord(formatter) == "{ 42 | 3.140000e+00 }");
    }

    SUBCASE("Appending another RecordFormatter") {
      RecordFormatter subFormatter;
      subFormatter << "Sub";
      subFormatter << "Formatter";

      RecordFormatter formatter;
      formatter << "Hello";
      formatter << subFormatter;

      std::ostringstream oss;
      oss << formatter;

      CHECK(formatRecord(formatter) == "{ Hello | { Sub | Formatter } }");
    }
  }
}

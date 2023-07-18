#include "utils/record_formatter.h"
#include "doctest.h"


TEST_CASE("RecordFormatter") {
  SUBCASE("Appending string") {
    RecordFormatter formatter;
    formatter << "Hello";
    formatter << "World";

    std::ostringstream oss;
    oss << formatter;

    CHECK(oss.str() == "{ Hello | World }");
  }

  SUBCASE("Appending integer and float") {
    RecordFormatter formatter;
    formatter << 42;
    formatter << 3.14f;

    std::ostringstream oss;
    oss << formatter;

    CHECK(oss.str() == "{ 42 | 3.140000e+00 }");
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

    CHECK(oss.str() == "{ Hello | { Sub | Formatter } }");
  }
}

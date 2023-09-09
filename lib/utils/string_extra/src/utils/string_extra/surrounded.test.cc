#include "utils/testing.h"
#include "utils/string_extra/surrounded.h"

TEST_CASE("surrounded") {
  CHECK_EQ(surrounded('"', "hello there"), "\"hello there\"");

  CHECK_EQ(surrounded('[', ']', "hello there"), "[hello there]");
  CHECK_EQ(surrounded("", "", "hello there"), "hello there");
  CHECK_EQ(surrounded("  ", "hello there"), "  hello there  ");
  CHECK_EQ(surrounded("abc: ", " :cba", "hello there"),
           "abc: hello there :cba");
  CHECK_EQ(surrounded("abc", "cba", ""), "abccba");
}

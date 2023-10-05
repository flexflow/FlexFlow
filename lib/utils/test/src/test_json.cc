#include "test/utils/doctest.h"
#include "utils/json.h"

static std::string str(json const &j) {
  std::stringstream ss;
  ss << j;
  return ss.str();
}

TEST_CASE("optional primitive") {
  SUBCASE("int") {
    CHECK(str(json(optional<int>())) == "null");

    CHECK(str(json(optional<int>(11))) == "11");

    // FIXME: There is a bug somewhere in the automatic JSON handling of an
    // optional<T> with a value in it that causes the code to go into an
    // infinite loop. This should be fixed ASAP.
    NOT_REACHABLE();
  }
}

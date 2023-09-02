#include "test/utils/all.h"
#include "utils/string.h"

TEST_CASE("surrounded") {
  CHECK_EQ(
    surrounded('"', "hello there"),
    "\"hello there\""
  );

  CHECK_EQ(
    surrounded('[', ']', "hello there"),
    "[hello there]"
  );
  CHECK_EQ(
    surrounded("", "", "hello there"),
    "hello there"
  );
  CHECK_EQ(
    surrounded("  ", "hello there"),
    "  hello there  "
  );
  CHECK_EQ(
    surrounded("abc: ", " :cba", "hello there"),
    "abc: hello there :cba"
  );
  CHECK_EQ(
    surrounded("abc", "cba", ""),
    "abccba"
  );
}

TEST_CASE("quoted") {
  CHECK_EQ(
    quoted("\"hello\" there \"my", '\\', '"'),
    "\\\"hello\\\" there \\\"my"
  );

  CHECK_EQ(
    quoted("a b c d e f g", '_', std::unordered_set{'b', 'd', 'e', 'f'}),
    "a _b c _d _e _f g"
  );

  CHECK_EQ(
    quoted("a b _ d", '_', std::unordered_set<char>{}),
    "a b __ d"
  );

  CHECK_EQ(
    quoted("", '_', 'a'),
    ""
  );

  CHECK_EQ(
    quoted("a b c d", 'a', 'a'),
    "aa b c d"
  );
    
  CHECK_EQ(
    quoted(std::string(2, 'a'), 'a', 'a'),
    std::string(4, 'a')
  );

  auto quote_a = [](std::string const &s) {
    return quoted(s, 'a');
  };

  CHECK_EQ(
    quote_a(quote_a(quote_a(std::string(3, 'a')))),
    std::string(3 * 2 * 2 * 2, 'a')
  );
}

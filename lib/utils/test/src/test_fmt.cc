#include "test/utils/all.h"
#include "utils/fmt.h"

using namespace ::FlexFlow::test_types;

TEST_CASE_TEMPLATE("std types are fmtable", 
                   T, 
                   std::vector<fmtable>,
                   std::list<fmtable>,
                   std::unordered_set<wb_hash_fmt>,
                   std::set<wb_fmt>,
                   std::unordered_map<wb_hash_fmt, wb_fmt>,
                   std::map<wb_fmt, wb_fmt>,
                   std::pair<fmtable, fmtable>,
                   std::tuple<fmtable, fmtable, fmtable, fmtable>
                   ) {
  STATIC_CHECK(is_fmtable_v<T>);
}

TEST_CASE("element_to_string") {
  CHECK(element_to_string(5) == "5");
  CHECK(element_to_string("hello \"your name\"") == "\"hello \\\"your name\\\"\"");
  CHECK(element_to_string('a') == "'a'");
  CHECK(element_to_string('\'') == "'\\\''");
}

TEST_CASE("std::vector formatting") {
  CHECK(fmt::to_string(std::vector<int>{1, 2, 3, 4}) == "[1, 2, 3, 4]");
  CHECK(fmt::to_string(std::vector<int>{}) == "[]");
  CHECK(fmt::to_string(std::vector<std::string>{"aa", "bb", "cc"}) == "[\"aa\", \"bb\", \"cc\"]");
}

TEST_CASE_TEMPLATE("equivalent formatting", 
                   T,
                   std::pair<std::vector<int>, std::list<int>>
                   std::pair<std::unordered_set<int>, std::set<int>>,
                   std::pair<std::unordered_map<int, std::string>, std::map<int, std::string>>
                   ) {
  using L = typename T::first_type;
  using R = typename T::second_type;
  rc::dc_check("matches formatting", [&](L l) {
    R r = {l.begin(), l.end()};
    CHECK(fmt::to_string(l) == fmt::to_string(r));
  });
}

TEST_CASE("equivalent formatting (std::pair and std::tuple)") {
  rc::dc_check("matches formatting", [&](std::pair<int, std::string> p) {
    std::tuple<int, std::string> t = { p.first, p.second };
    CHECK(fmt::to_string(p) == fmt::to_string(t));
  });
}

TEST_CASE("std::unordered_set formatting") {
  CHECK(fmt::to_string(std::unordered_set<int>{1, 2, 3, 4}) == "{1, 2, 3, 4}");
  CHECK(fmt::to_string(std::unordered_set<int>{}) == "{}");
}

TEST_CASE("std::unordered_map formatting") {
  std::unordered_map<int, int> m = {
    { 1, "hello" },
    { 5, "yes" },
    { 1000, "" }
  };
  CHECK(fmt::to_string(m) == "{<1, \"hello\">, <5, \"yes\">, <1000, \"\">}");
}

TEST_CASE("std::tuple formatting") {
  CHECK(fmt::to_string(std::tuple{1, "hi", 5.3}) == "<1, \"hi\", 5.3>");
}

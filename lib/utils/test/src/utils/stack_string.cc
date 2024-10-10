#include "utils/stack_string.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("StackStringConstruction", T, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    SUBCASE("DefaultConstruction") {
      StackString str;
      CHECK(str.size() == 0);
      CHECK(str.length() == 0);
      CHECK(static_cast<std::string>(str) == "");
    }

    SUBCASE("CStringConstruction") {
      char const *cstr = "Hello";
      StackString str(cstr);
      CHECK(str.size() == 5);
      CHECK(str.length() == 5);
      CHECK(static_cast<std::string>(str) == "Hello");
    }

    SUBCASE("ShortCStringConstruction") {
      char const *cstr = "CMU";
      StackString str(cstr);
      CHECK(str.size() == 3);
      CHECK(str.length() == 3);
      CHECK(static_cast<std::string>(str) == "CMU");
    }

    SUBCASE("StdStringConstruction") {
      std::basic_string<T> stdStr = "World";
      StackString str(stdStr);
      CHECK(str.size() == 5);
      CHECK(str.length() == 5);
      CHECK(static_cast<std::string>(str) == "World");
    }
  }

  TEST_CASE_TEMPLATE("StackStringComparison", T, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    StackString str1{"abc"};
    StackString str2{"def"};
    StackString str3{"abc"};

    CHECK(str1 == str1);
    CHECK(str1 == str3);
    CHECK(str1 != str2);
    CHECK(str2 != str3);
    CHECK(str1 < str2);
  }

  TEST_CASE_TEMPLATE("StackStringSize", T, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    SUBCASE("EmptyString") {
      StackString str;
      CHECK(str.size() == 0);
      CHECK(str.length() == 0);
    }

    SUBCASE("NonEmptyString") {
      StackString str{"Hello"};
      CHECK(str.size() == 5);
      CHECK(str.length() == 5);
    }
  }

  TEST_CASE_TEMPLATE("StackStringConversion", T, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    StackString str{"Hello"};
    std::string stdStr = static_cast<std::string>(str);
    CHECK(stdStr == "Hello");
  }

  TEST_CASE("Arbitrary<stack_string>") {
    constexpr std::size_t MAXSIZE = 10;
    RC_SUBCASE([&](stack_string<MAXSIZE> const &s) {
      RC_ASSERT(s.size() <= MAXSIZE);
    });
  }
}

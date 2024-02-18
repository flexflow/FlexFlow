#include "utils/testing.h"
#include "utils/bidict/algorithms/transform.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform_keys(bidict<K, V> const &, F const &)") {
    SUBCASE("type-preserving") {
      bidict<int, std::string> b;
      b.equate(1, "one");
      b.equate(2, "two");

      bidict<int, std::string> correct;
      correct.equate(2, "one");
      correct.equate(3, "two");

      bidict<int, std::string> result = transform_keys(b, [](int k) { return k + 1; });
      
      CHECK(result == correct);
    }
    
    SUBCASE("type-changing") {
      bidict<int, std::string> b;
      b.equate(1, "one");
      b.equate(2, "two");

      bidict<std::string, std::string> correct;
      correct.equate("1", "one");
      correct.equate("2", "two");

      bidict<std::string, std::string> result = transform_keys(b, [](int k) { return std::to_string(k); });

      CHECK(result == correct);
    }
  }

  TEST_CASE("transform_values(bidict<K, V> const &, F const &)") {
    SUBCASE("type-preserving") {
      bidict<int, std::string> b;
      b.equate(1, "one");
      b.equate(2, "two");

      bidict<int, std::string> correct;
      correct.equate(1, "one plus one");
      correct.equate(2, "two plus one");

      bidict<int, std::string> result = transform_values(b, [](std::string const &v) { return v + " plus one"; });
    
      CHECK(result == correct);
    }

    SUBCASE("type-changing") {
      bidict<int, std::string> b;
      b.equate(1, "aa");
      b.equate(2, "aaaaa");

      bidict<int, size_t> correct;
      correct.equate(1, 2);
      correct.equate(2, 5);

      bidict<int, size_t> result = transform_values(b, [](std::string const &v) { return v.size(); });
    
      CHECK(result == correct);
    }
  }
}

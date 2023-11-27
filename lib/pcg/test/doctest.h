#include "doctest/doctest.h"
#include "utils/containers.h"
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace doctest {

template <typename InputIt, typename Stringifiable = std::string>
std::string
    doctest_print_container(InputIt first,
                            InputIt last,
                            std::string const &open,
                            std::string const &delimiter,
                            std::string const &close,
                            std::function<Stringifiable(InputIt)> const &f) {
  if (first == last) {
    return open + "(empty)" + close;
  } else {
    return open + FlexFlow::join_strings(first, last, delimiter, f) + close;
  }
}

template <typename InputIt>
std::string doctest_print_container(InputIt first,
                                    InputIt last,
                                    std::string const &open,
                                    std::string const &delimiter,
                                    std::string const &close) {
  return doctest_print_container<InputIt, decltype(*first)>(
      first, last, open, delimiter, close, [](InputIt ref) { return *ref; });
}

template <typename Container>
std::string doctest_print_container(Container const &c,
                                    std::string const &open,
                                    std::string const &delimiter,
                                    std::string const &close) {
  return doctest_print_container<decltype(c.cbegin())>(
      c.cbegin(), c.cend(), open, delimiter, close);
}

template <typename T>
struct StringMaker<std::unordered_set<T>> {
  static String convert(std::unordered_set<T> const &s) {
    return doctest_print_container(s, "{ ", ", ", " }").c_str();
  }
};

template <typename K, typename V>
struct StringMaker<std::unordered_map<K, V>> {
  static String convert(std::unordered_map<K, V> const &m) {
    std::unordered_set<std::string> entries;
    for (auto const &kv : m) {
      std::ostringstream oss;
      oss << toString(kv.first) << " -> " << toString(kv.second);
      entries.insert(oss.str());
    }
    return toString(entries);
  }
};

template <typename T>
struct StringMaker<std::vector<T>> {
  static String convert(std::vector<T> const &vec) {
    return doctest_print_container(vec, "[ ", ", ", " ]").c_str();
  }
};
} // namespace doctest

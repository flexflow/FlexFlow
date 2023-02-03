#include "doctest/doctest.h"
#include <unordered_set>
#include <unordered_map>
#include <sstream>

namespace doctest
{
template <typename T>
struct StringMaker<std::unordered_set<T>> {
  static String convert(std::unordered_set<T> const &s) {
    if (s.empty()) {
      return "{ (empty) }";
    }

    std::ostringstream oss;
    
    oss << "{ ";
    bool first = true;
    for (T const &n : s) {
      if (!first) {
        oss << ", ";
      }
      oss << n;
      first = false;
    }
    oss << " }";
    return oss.str().c_str();
  }
};

template <typename K, typename V>
struct StringMaker<std::unordered_map<K, V>> {
  static String convert(std::unordered_map<K, V> const &m) {
    if (m.empty()) {
      return "{ (empty) }";
    }

    std::ostringstream oss;

    oss << "{ ";
    bool first = true;
    for (auto const &kv : m) {
      if (!first) {
        oss << ", ";
      }
      oss << kv.first << " -> " << toString(kv.second);
      first = false;
    }
    oss << " }";
    return oss.str().c_str();
  }
};
}

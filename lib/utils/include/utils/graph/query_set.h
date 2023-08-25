#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_QUERY_SET_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_QUERY_SET_H

#include "utils/bidict.h"
#include "utils/containers.decl"
#include "utils/exception.h"
#include "utils/optional.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
struct query_set {
  query_set() = delete;
  query_set(T const &query) : query({query}) {}

  query_set(std::unordered_set<T> const &query) : query(query) {}

  query_set(optional<std::unordered_set<T>> const &query) : query(query) {}

  query_set(std::initializer_list<T> const &l)
      : query_set(std::unordered_set<T>{l}) {}

  friend bool operator==(query_set const &lhs, query_set const &rhs) {
    return lhs.value == rhs.value;
  }

  friend bool operator!=(query_set const &lhs, query_set const &rhs) {
    return lhs.value != rhs.value;
  }

  friend bool operator<(query_set const &lhs, query_set const &rhs) {
    return lhs.value < rhs.value;
  }

  friend bool is_matchall(query_set const &q) {
    return !q.query.has_value();
  }

  friend std::unordered_set<T> allowed_values(query_set const &q) {
    assert(!is_matchall(q));
    return q.query.value();
  }

  static query_set<T> matchall() {
    return {nullopt};
  }

private:
  optional<std::unordered_set<T>> query;
};

template <typename T>
query_set<T> matchall() {
  return query_set<T>::matchall();
}

template <typename T>
bool includes(query_set<T> const &q, T const &v) {
  return is_matchall(q) || contains(allowed_values(q), v);
}

template <typename T, typename C>
std::unordered_set<T> apply_query(query_set<T> const &q, C const &c) {
  if (is_matchall(q)) {
    return unique(c);
  }

  return filter(unique(c), [&](T const &t) { return includes(q, t); });
}

template <typename C,
          typename K = typename C::key_type,
          typename V = typename C::mapped_type>
std::unordered_map<K, V> query_keys(query_set<K> const &q, C const &m) {
  if (is_matchall(q)) {
    return m;
  }
  return filter_keys(m, [&](K const &key) { return includes(q, key); });
}

template <typename C,
          typename K = typename C::key_type,
          typename V = typename C::mapped_type>
std::unordered_map<K, V> query_values(query_set<V> const &q, C const &m) {
  if (is_matchall(q)) {
    return m;
  }
  return filter_values(m, [&](V const &value) { return includes(q, value); });
}

template <typename T>
query_set<T> query_intersection(query_set<T> const &lhs,
                                query_set<T> const &rhs) {
  if (is_matchall(lhs)) {
    return rhs;
  } else if (is_matchall(rhs)) {
    return lhs;
  } else {
    return intersection(allowed_values(lhs), allowed_values(rhs));
  }
}

template <typename T>
query_set<T> query_union(query_set<T> const &lhs, query_set<T> const &rhs) {
  if (is_matchall(lhs) || is_matchall(rhs)) {
    return query_set<T>::matchall();
  } else {
    return set_union(allowed_values(lhs), allowed_values(rhs));
  }
}

} // namespace FlexFlow

namespace fmt {

template <typename T>
struct formatter<::FlexFlow::query_set<T>> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::query_set<T> const &q, FormatContext &ctx)
      -> decltype(ctx.out()) {
    std::string result;
    if (is_matchall(q)) {
      result = "(all)";
    } else {
      result = fmt::format("query_set({})", allowed_values(q));
    }
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

#include "utils/containers.h"

#endif

#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_QUERY_SET_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_QUERY_SET_H

#include "utils/bidict/bidict.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/containers/filter_keys.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/set_of.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/fmt/set.h"
#include "utils/hash-utils.h"
#include "utils/hash/set.h"
#include "utils/json/optional.h"
#include "utils/optional.h"
#include <optional>
#include <set>

namespace FlexFlow {

template <typename T>
struct query_set {
  query_set() = delete;

  static query_set<T> matchall() {
    std::optional<std::set<T>> query_val = std::nullopt;
    return query_set<T>{
        query_val,
    };
  }

  static query_set<T> match_none() {
    std::set<T> to_match = {};

    return query_set<T>{
        std::optional<std::set<T>>{to_match},
    };
  }

  static query_set<T> match_values_in(std::set<T> const &values) {
    return query_set<T>{
        std::optional<std::set<T>>{values},
    };
  }

  static query_set<T> match_single_value(T const &val) {
    std::set<T> vals = {val};

    return query_set<T>::match_values_in(vals);
  }

  friend bool operator==(query_set const &lhs, query_set const &rhs) {
    return lhs.query == rhs.query;
  }

  friend bool operator!=(query_set const &lhs, query_set const &rhs) {
    return lhs.query != rhs.query;
  }

  friend bool operator<(query_set const &lhs, query_set const &rhs) {
    return lhs.query < rhs.query;
  }

  friend bool is_matchall(query_set const &q) {
    return !q.query.has_value();
  }

  friend std::set<T> const &allowed_values(query_set const &q) {
    return assert_unwrap(q.value());
  }

  std::optional<std::set<T>> const &value() const {
    return this->query;
  }

private:
  explicit query_set(std::optional<std::set<T>> const &query) : query(query) {}

private:
  std::optional<std::set<T>> query;
};

template <typename T>
std::string format_as(query_set<T> const &q) {
  if (is_matchall(q)) {
    return "(all)";
  } else {
    return fmt::format(FMT_STRING("query_set({})"), allowed_values(q));
  }
}

template <typename T>
struct delegate_ostream_operator<query_set<T>> : std::true_type {};

template <typename T>
query_set<T> matchall() {
  return query_set<T>::matchall();
}

template <typename T>
bool includes(query_set<T> const &q, T const &v) {
  return is_matchall(q) || contains(allowed_values(q), v);
}

template <typename T, typename C>
std::set<T> apply_query(query_set<T> const &q, C const &c) {
  return apply_query(q, set_of(c));
}

template <typename T>
std::set<T> apply_query(query_set<T> const &q, std::set<T> const &c) {
  if (is_matchall(q)) {
    return c;
  }

  return set_intersection(allowed_values(q), c);
}

template <typename C,
          typename K = typename C::key_type,
          typename V = typename C::mapped_type>
std::map<K, V> query_keys(query_set<K> const &q, C const &m) {
  if (is_matchall(q)) {
    return m;
  }
  return filter_keys(m, [&](K const &key) { return includes(q, key); });
}

template <typename C,
          typename K = typename C::key_type,
          typename V = typename C::mapped_type>
std::map<K, V> query_values(query_set<V> const &q, C const &m) {
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
    return query_set<T>::match_values_in(
        set_of(set_intersection(allowed_values(lhs), allowed_values(rhs))));
  }
}

template <typename T>
query_set<T> query_union(query_set<T> const &lhs, query_set<T> const &rhs) {
  if (is_matchall(lhs) || is_matchall(rhs)) {
    return query_set<T>::matchall();
  } else {
    return query_set<T>::match_values_in(
        set_of(set_union(allowed_values(lhs), allowed_values(rhs))));
  }
}

template <typename T>
void to_json(nlohmann::json &j, query_set<T> const &q) {
  j["__type"] = "query_set";
  j["value"] = q.value();
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::query_set<T>> {
  size_t operator()(::FlexFlow::query_set<T> const &q) const {
    return ::FlexFlow::get_std_hash(q.value());
  }
};

} // namespace std

#endif

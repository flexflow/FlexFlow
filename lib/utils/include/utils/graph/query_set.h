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
#include "utils/hash/tuple.h"
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
        false,
    };
  }

  static query_set<T> match_none() {
    std::set<T> to_match = {};

    return query_set<T>{
        std::optional<std::set<T>>{to_match},
        false,
    };
  }

  static query_set<T> match_values_in(std::set<T> const &values) {
    return query_set<T>{
        std::optional<std::set<T>>{values},
        false,
    };
  }

  static query_set<T> match_except_values_in(std::set<T> const &values) {
    return query_set<T>{
        std::optional<std::set<T>>{values},
        true,
    };
  }

  static query_set<T> match_single_value(T const &val) {
    std::set<T> vals = {val};

    return query_set<T>::match_values_in(vals);
  }

  static query_set<T> match_except_single_value(T const &val) {
    std::set<T> vals = {val};

    return query_set<T>::match_except_values_in(vals);
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

  friend std::set<T> const &allowed_values(query_set const &q) {
    ASSERT(!q.is_negated());
    return assert_unwrap(q.query);
  }

  friend std::set<T> const &disallowed_values(query_set const &q) {
    ASSERT(q.is_negated());
    return assert_unwrap(q.query);
  }

  friend bool is_matchnone(query_set const &q) {
    if (q.is_negated()) {
      return !q.query.has_value();
    } else {
      return q.query.has_value() && allowed_values(q).empty();
    }
  }

  friend bool is_matchall(query_set const &q) {
    if (q.is_negated()) {
      return q.query.has_value() && disallowed_values(q).empty();
    } else {
      return !q.query.has_value();
    }
  }

  bool is_negated() const {
    return this->m_negated;
  }

  query_set<T> negated() const {
    return query_set<T>{
        this->query,
        !this->m_negated,
    };
  }

  std::optional<std::set<T>> const &raw_query() const {
    return this->query;
  }

private:
  explicit query_set(std::optional<std::set<T>> const &query, bool negated)
      : query(query), m_negated(negated) {}

private:
  std::optional<std::set<T>> query;
  bool m_negated;

private:
  std::tuple<decltype(query) const &, decltype(m_negated) const &> tie() const {
    return std::tie(this->query, this->m_negated);
  }

  friend struct ::std::hash<query_set>;
};

template <typename T>
std::string format_as(query_set<T> const &q) {
  if (is_matchall(q)) {
    return "(all)";
  }

  if (is_matchnone(q)) {
    return "(none)";
  }

  if (q.is_negated()) {
    return fmt::format(FMT_STRING("query_set(not {})"), disallowed_values(q));
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
  if (is_matchall(q)) {
    return true;
  }

  if (is_matchnone(q)) {
    return false;
  }

  if (q.is_negated()) {
    return !contains(disallowed_values(q), v);
  } else {
    return contains(allowed_values(q), v);
  }
}

template <typename T, typename C>
std::set<T> apply_query(query_set<T> const &q, C const &c) {
  if (is_matchall(q)) {
    return set_of(c);
  }

  if (is_matchnone(q)) {
    return std::set<T>{};
  }

  if (q.is_negated()) {
    std::set<T> const &disallowed = disallowed_values(q);

    std::set<T> result;

    std::set_difference(c.cbegin(),
                        c.cend(),
                        disallowed.cbegin(),
                        disallowed.cend(),
                        std::inserter(result, result.begin()));

    return result;
  } else {
    std::set<T> const &allowed = allowed_values(q);

    std::set<T> result;

    std::set_intersection(c.cbegin(),
                          c.cend(),
                          allowed.cbegin(),
                          allowed.cend(),
                          std::inserter(result, result.begin()));

    return result;
  }
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
        set_intersection(allowed_values(lhs), allowed_values(rhs)));
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
  j["query"] = q.raw_query();
  j["is_negated"] = q.is_negated();
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::query_set<T>> {
  size_t operator()(::FlexFlow::query_set<T> const &q) const {
    return ::FlexFlow::get_std_hash(q.tie());
  }
};

} // namespace std

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_H

#include "utils/containers/keys.h"
#include "utils/containers/require_same.h"
#include "utils/containers/set_of.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"
#include "utils/fmt/map.h"
#include "utils/fmt/set.h"
#include "utils/hash-utils.h"
#include "utils/hash/map.h"
#include "utils/hash/tuple.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include "utils/nonempty_set/nonempty_set.h"
#include "utils/not_implemented.h"
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <rapidcheck.h>

namespace FlexFlow {

template <typename L, typename R>
struct ManyToOne {
public:
  ManyToOne() : m_l_to_r(), m_r_to_l() {}

  template <typename It>
  ManyToOne(It start, It end) : ManyToOne() {
    for (; start < end; start++) {
      ASSERT(start->first.size() > 0);
      for (L const &l : start->first) {
        this->insert(std::pair<L, R>{l, start->second});
      }
    }
  }

  ManyToOne(std::initializer_list<std::pair<std::initializer_list<L>, R>> const
                &l_to_r)
      : ManyToOne(l_to_r.begin(), l_to_r.end()) {}

  bool operator==(ManyToOne const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(ManyToOne const &other) const {
    return this->tie() != other.tie();
  }

  void insert(std::pair<L, R> const &p) {
    L l = p.first;
    R r = p.second;

    std::optional<R> found_r = try_at(this->m_l_to_r, l);

    if (!found_r.has_value()) {
      this->m_l_to_r.insert({l, r});

      if (contains_key(this->m_r_to_l, r)) {
        this->m_r_to_l.at(r).insert(l);
      } else {
        this->m_r_to_l.insert({r, nonempty_set{{l}}});
      }
    } else if (found_r.value() == r) {
      return;
    } else {
      PANIC("Existing mapping found for left value {}: tried to map to right "
            "value {}, but is already bound to right value {}",
            l,
            r,
            found_r.value());
    }
  }

  bool contains_l(L const &l) const {
    return contains_key(this->m_l_to_r, l);
  }

  bool contains_r(R const &r) const {
    return contains_key(this->m_r_to_l, r);
  }

  R const &at_l(L const &l) const {
    return this->m_l_to_r.at(l);
  }

  nonempty_set<L> const &at_r(R const &r) const {
    return this->m_r_to_l.at(r);
  }

  std::set<L> left_values() const {
    return keys(this->m_l_to_r);
  }

  std::set<nonempty_set<L>> left_groups() const {
    return set_of(values(this->m_r_to_l));
  }

  std::set<R> right_values() const {
    return keys(this->m_r_to_l);
  }

  std::map<L, R> const &l_to_r() const {
    return this->m_l_to_r;
  }

  std::map<R, nonempty_set<L>> const &r_to_l() const {
    return this->m_r_to_l;
  }

  bool empty() const {
    return require_same(this->m_l_to_r.empty(), this->m_r_to_l.empty());
  }

private:
  std::map<L, R> m_l_to_r;
  std::map<R, nonempty_set<L>> m_r_to_l;

private:
  std::tuple<decltype(m_l_to_r) const &, decltype(m_r_to_l) const &>
      tie() const {
    return std::tie(this->m_l_to_r, this->m_r_to_l);
  }

  friend struct std::hash<ManyToOne<L, R>>;
};

template <typename L, typename R>
std::map<nonempty_set<L>, R> format_as(ManyToOne<L, R> const &m) {
  std::map<nonempty_set<L>, R> result;

  for (R const &r : m.right_values()) {
    result.insert({m.at_r(r), r});
  }

  return result;
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, ManyToOne<L, R> const &m) {
  return (s << fmt::to_string(m));
}

template <typename L, typename R>
std::set<std::pair<L, R>>
    unstructured_relation_from_many_to_one(ManyToOne<L, R> const &many_to_one) {
  return set_of(many_to_one.l_to_r());
}

template <typename L, typename R>
ManyToOne<L, R> many_to_one_from_unstructured_relation(
    std::set<std::pair<L, R>> const &relation) {
  ManyToOne<L, R> result;
  for (auto const &lr : relation) {
    result.insert(lr);
  }
  return result;
}

} // namespace FlexFlow

namespace nlohmann {

template <typename L, typename R>
struct adl_serializer<::FlexFlow::ManyToOne<L, R>> {
  static ::FlexFlow::ManyToOne<L, R> from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(L);
    CHECK_IS_JSON_DESERIALIZABLE(R);

    std::set<std::pair<L, R>> s = j;

    return ::FlexFlow::many_to_one_from_unstructured_relation(s);
  }

  static void to_json(json &j, ::FlexFlow::ManyToOne<L, R> const &m) {
    CHECK_IS_JSON_SERIALIZABLE(L);
    CHECK_IS_JSON_SERIALIZABLE(R);

    j = ::FlexFlow::set_of(
        ::FlexFlow::unstructured_relation_from_many_to_one(m));
  }
};

} // namespace nlohmann

namespace rc {

template <typename L, typename R>
struct Arbitrary<::FlexFlow::ManyToOne<L, R>> {
  static Gen<::FlexFlow::ManyToOne<L, R>> arbitrary() {
    NOT_IMPLEMENTED();
  }
};

} // namespace rc

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::ManyToOne<L, R>> {
  size_t operator()(::FlexFlow::ManyToOne<L, R> const &m) {
    return ::FlexFlow::get_std_hash(m.tie());
  }
};

} // namespace std

#endif

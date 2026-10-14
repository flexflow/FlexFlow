#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_H

#include "utils/containers/generate_map.h"
#include "utils/containers/items.h"
#include "utils/containers/keys.h"
#include "utils/containers/require_same.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"
#include "utils/fmt/map.h"
#include "utils/fmt/set.h"
#include "utils/hash-utils.h"
#include "utils/hash/map.h"
#include "utils/hash/set.h"
#include "utils/hash/tuple.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include "utils/nonempty_set/nonempty_set.h"
#include "utils/not_implemented.h"
#include <fmt/format.h>
#include <map>
#include <nlohmann/json.hpp>
#include <rapidcheck.h>
#include <set>

namespace FlexFlow {

template <typename L, typename R>
struct OneToMany {
public:
  OneToMany() : m_l_to_r(), m_r_to_l() {}

  template <typename It>
  OneToMany(It start, It end) : OneToMany() {
    for (; start < end; start++) {
      ASSERT(start->second.size() > 0);
      for (R const &r : start->second) {
        this->insert(std::pair<L, R>{start->first, r});
      }
    }
  }

  OneToMany(std::initializer_list<std::pair<L, std::initializer_list<R>>> const
                &l_to_r)
      : OneToMany(l_to_r.begin(), l_to_r.end()) {}

  bool operator==(OneToMany const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(OneToMany const &other) const {
    return this->tie() != other.tie();
  }

  bool operator<(OneToMany const &other) const {
    return this->tie() < other.tie();
  }

  bool operator<=(OneToMany const &other) const {
    return this->tie() <= other.tie();
  }

  bool operator>(OneToMany const &other) const {
    return this->tie() > other.tie();
  }

  bool operator>=(OneToMany const &other) const {
    return this->tie() >= other.tie();
  }

  void insert(std::pair<L, R> const &p) {
    L l = p.first;
    R r = p.second;

    std::optional<L> found_l = try_at(this->m_r_to_l, r);

    if (!found_l.has_value()) {
      this->m_r_to_l.insert({r, l});

      if (contains_key(this->m_l_to_r, l)) {
        this->m_l_to_r.at(l).insert(r);
      } else {
        this->m_l_to_r.insert({l, nonempty_set{{r}}});
      }
    } else if (found_l.value() == l) {
      return;
    } else {
      PANIC("Existing mapping found for right value {}: tried to map "
            "to left value {}, but is already bound to left value {}",
            r,
            l,
            found_l.value());
    }
  }

  std::set<std::pair<L, R>> relation() const {
    return transform(items(this->m_r_to_l),
                     [](std::pair<R, L> const &p) -> std::pair<L, R> {
                       return {p.second, p.first};
                     });
  }

  nonempty_set<R> const &at_l(L const &l) const {
    return this->m_l_to_r.at(l);
  }

  L const &at_r(R const &r) const {
    return this->m_r_to_l.at(r);
  }

  std::set<L> left_values() const {
    return keys(this->m_l_to_r);
  }

  std::set<R> right_values() const {
    return keys(this->m_r_to_l);
  }

  std::set<nonempty_set<R>> right_groups() const {
    return set_of(values(this->m_l_to_r));
  }

  std::map<L, nonempty_set<R>> const &l_to_r() const {
    return this->m_l_to_r;
  }

  std::map<R, L> const &r_to_l() const {
    return this->m_r_to_l;
  }

  bool empty() const {
    return require_same(this->m_l_to_r.empty(), this->m_r_to_l.empty());
  }

private:
  std::map<L, nonempty_set<R>> m_l_to_r;
  std::map<R, L> m_r_to_l;

private:
  std::tuple<decltype(m_l_to_r) const &, decltype(m_r_to_l) const &>
      tie() const {
    return std::tie(this->m_l_to_r, this->m_r_to_l);
  }

  friend struct std::hash<OneToMany<L, R>>;
};

template <typename L, typename R>
std::map<L, nonempty_set<R>> format_as(OneToMany<L, R> const &m) {
  return generate_map(m.left_values(), [&](L const &l) { return m.at_l(l); });
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, OneToMany<L, R> const &m) {
  return (s << fmt::to_string(m));
}

template <typename L, typename R>
std::set<std::pair<L, R>>
    unstructured_relation_from_one_to_many(OneToMany<L, R> const &one_to_many) {
  return transform(set_of(one_to_many.r_to_l()),
                   [](std::pair<R, L> const &rl) -> std::pair<L, R> {
                     return std::pair{rl.second, rl.first};
                   });
}

template <typename L, typename R>
OneToMany<L, R> one_to_many_from_unstructured_relation(
    std::set<std::pair<L, R>> const &rel) {
  OneToMany<L, R> result;
  for (auto const &lr : rel) {
    result.insert(lr);
  }
  return result;
}

} // namespace FlexFlow

namespace nlohmann {

template <typename L, typename R>
struct adl_serializer<::FlexFlow::OneToMany<L, R>> {
  static ::FlexFlow::OneToMany<L, R> from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(L);
    CHECK_IS_JSON_DESERIALIZABLE(R);

    std::set<std::pair<L, R>> s = j;

    return ::FlexFlow::one_to_many_from_unstructured_relation(s);
  }

  static void to_json(json &j, ::FlexFlow::OneToMany<L, R> const &m) {
    CHECK_IS_JSON_SERIALIZABLE(L);
    CHECK_IS_JSON_SERIALIZABLE(R);

    j = ::FlexFlow::set_of(
        ::FlexFlow::unstructured_relation_from_one_to_many(m));
  }
};

} // namespace nlohmann

namespace rc {

template <typename L, typename R>
struct Arbitrary<::FlexFlow::OneToMany<L, R>> {
  static Gen<::FlexFlow::OneToMany<L, R>> arbitrary() {
    NOT_IMPLEMENTED();
  }
};

} // namespace rc

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::OneToMany<L, R>> {
  size_t operator()(::FlexFlow::OneToMany<L, R> const &m) {
    return ::FlexFlow::get_std_hash(m.tie());
  }
};

} // namespace std

#endif

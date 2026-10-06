#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_BIDICT_H

#include "utils/check_fmtable.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/require_same.h"
#include "utils/containers/set_of.h"
#include "utils/containers/unordered_map_from_map.h"
#include "utils/containers/values.h"
#include "utils/fmt/map.h"
#include "utils/hash/map.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include <cassert>
#include <nlohmann/json.hpp>
#include <optional>
#include <rapidcheck.h>

namespace FlexFlow {

template <typename L, typename R>
struct bidict {
  bidict() : fwd_map{}, bwd_map{} {}

  bidict(std::initializer_list<std::pair<L, R>> init)
      : bidict(init.begin(), init.end()) {}

  template <typename InputIt>
  bidict(InputIt first, InputIt last) {
    for (auto it = first; it != last; it++) {
      this->equate(it->first, it->second);
    }
  }

  bool contains(L const &l, R const &r) const {
    return this->contains_l(l) && this->at_l(l) == r;
  }

  bool contains_l(L const &l) const {
    return fwd_map.find(l) != fwd_map.end();
  }

  bool contains_r(R const &r) const {
    return bwd_map.find(r) != bwd_map.end();
  }

  void erase_l(L const &l) {
    if (this->contains_l(l)) {
      R r = this->at_l(l);
      this->fwd_map.erase(l);
      this->bwd_map.erase(r);
    }
  }

  void erase_r(R const &r) {
    if (this->contains_r(r)) {
      L l = this->at_r(r);
      this->fwd_map.erase(l);
      this->bwd_map.erase(r);
    }
  }

  bool try_equate(L const &l, R const &r) {
    auto [fwd_it, fwd_did_insert] = fwd_map.insert({l, r});
    if (!fwd_did_insert && fwd_it->second != r) {
      return false;
    }

    auto [bwd_it, bwd_did_insert] = bwd_map.insert({r, l});
    if (!bwd_did_insert && bwd_it->second != l) {
      if (fwd_did_insert) {
        fwd_map.erase(fwd_it);
      }

      return false;
    }

    return true;
  }

  void equate(L const &l, R const &r) {
    ASSERT(this->contains_l(l) == this->contains_r(r));

    if (this->contains_l(l)) {
      ASSERT(this->at_l(l) == r);
    } else {
      fwd_map.insert({l, r});
      bwd_map.insert({r, l});
    }
  }

  void equate(std::pair<L, R> const &lr) {
    this->equate(lr.first, lr.second);
  }

  bool operator==(bidict<L, R> const &other) const {
    return require_same((this->fwd_map == other.fwd_map),
                        (this->bwd_map == other.bwd_map));
  }

  bool operator!=(bidict<L, R> const &other) const {
    return require_same((this->fwd_map != other.fwd_map),
                        (this->bwd_map != other.bwd_map));
  }

  R const &at_l(L const &l) const {
    DEBUG_ASSERT(contains_key(this->fwd_map, l));
    return fwd_map.at(l);
  }

  L const &at_r(R const &r) const {
    DEBUG_ASSERT(contains_key(this->bwd_map, r));
    return bwd_map.at(r);
  }

  std::set<L> left_values() const {
    return keys(this->fwd_map);
  }

  std::set<R> right_values() const {
    return keys(this->bwd_map);
  }

  std::size_t size() const {
    assert(fwd_map.size() == bwd_map.size());
    return fwd_map.size();
  }

  bool empty() const {
    return this->size() == 0;
  }

  using const_iterator = typename std::map<L, R>::const_iterator;
  using value_type = std::pair<L, R>;
  using reference = value_type &;
  using const_reference = value_type const &;
  using key_type = L;
  using mapped_type = R;
  /* struct const_iterator { */
  /*   using iterator_category = std::forward_iterator_tag; */
  /*   using difference_type = std::size_t; */
  /*   using value_type = std::pair<L, R>; */
  /*   using pointer = std::pair<L, R> const *; */
  /*   using reference = std::pair<L, R> const &; */

  /*   explicit const_iterator(typename std::map<tl::optional<L>,
   * tl::optional<R>>::const_iterator); */

  /*   reference operator*() const { */
  /*     this->current = {this->it->first.value(), this->it->second.value()}; */
  /*     return this->current.value(); */
  /*   } */
  /*   pointer operator->() const { */
  /*     return &this->operator*(); */
  /*   } */

  /*   const_iterator& operator++() { */
  /*     ++this->it; */
  /*     return *this; */
  /*   } */
  /*   const_iterator operator++(int) { */
  /*     auto tmp = *this; */
  /*     ++(*this); */
  /*     return tmp; */
  /*   } */

  /*   bool operator==(const_iterator const &other) const { */
  /*     return this->it == other.it; */
  /*   } */
  /*   bool operator!=(const_iterator const &other) const { */
  /*     return this->it != other.it; */
  /*   } */
  /* private: */
  /*   mutable tl::optional<std::pair<L, R>> current; */
  /*   typename std::map<tl::optional<L>,
   * tl::optional<R>>::const_iterator it; */
  /* }; */

  /* const_iterator cbegin() const { */
  /*   return const_iterator(this->fwd_map.cbegin()); */
  /* } */

  /* const_iterator begin() const { */
  /*   return this->cbegin(); */
  /* } */

  /* const_iterator cend() const { */
  /*   return const_iterator(this->fwd_map.cend()); */
  /* } */

  /* const_iterator end() const { */
  /*   return this->cend(); */
  /* } */

  const_iterator cbegin() const {
    return this->fwd_map.cbegin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator cend() const {
    return this->fwd_map.cend();
  }

  const_iterator end() const {
    return this->cend();
  }

  bidict<R, L> reversed() const {
    return bidict<R, L>(bwd_map, fwd_map);
  }

  operator std::map<L, R> const &() const {
    return this->fwd_map;
  }

  operator std::unordered_map<L, R>() const {
    return unordered_map_from_map(this->fwd_map);
  }

  std::map<L, R> const &as_map() const {
    return this->fwd_map;
  }

  std::unordered_map<L, R> as_unordered_map() const {
    return unordered_map_from_map(this->fwd_map);
  }

  std::map<L, R> const &l_to_r() const {
    return this->fwd_map;
  }

  std::map<R, L> const &r_to_l() const {
    return this->bwd_map;
  }

  bidict(std::map<L, R> const &fwd_map, std::map<R, L> const &bwd_map)
      : fwd_map(fwd_map), bwd_map(bwd_map) {
    this->check_invariants();
  }

  bool operator<(bidict<L, R> const &other) const {
    return this->fwd_map < other.fwd_map;
  }

  bool operator<=(bidict<L, R> const &other) const {
    return this->fwd_map <= other.fwd_map;
  }

  bool operator>(bidict<L, R> const &other) const {
    return this->fwd_map > other.fwd_map;
  }

  bool operator>=(bidict<L, R> const &other) const {
    return this->fwd_map >= other.fwd_map;
  }

private:
  void check_invariants() const {
    std::set<L> fwd_l_vals = keys(this->fwd_map);
    std::set<L> bwd_l_vals = set_of(values(this->bwd_map));

    std::set<R> bwd_r_vals = keys(this->bwd_map);
    std::set<R> fwd_r_vals = set_of(values(this->fwd_map));

    ASSERT(fwd_l_vals == bwd_l_vals);
    ASSERT(fwd_r_vals == bwd_r_vals);

    for (L const &l : fwd_l_vals) {
      ASSERT(bwd_map.at(fwd_map.at(l)) == l);
    }
  }

  friend struct bidict<R, L>;

  std::map<L, R> fwd_map;
  std::map<R, L> bwd_map;
};

template <typename L, typename R>
std::map<L, R> format_as(bidict<L, R> const &b) {
  return b;
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, bidict<L, R> const &b) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return s << fmt::to_string(b);
}

} // namespace FlexFlow

namespace nlohmann {

template <typename L, typename R>
struct adl_serializer<::FlexFlow::bidict<L, R>> {
  static ::FlexFlow::bidict<L, R> from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(L);
    CHECK_IS_JSON_DESERIALIZABLE(R);

    std::map<L, R> m = j;

    ::FlexFlow::bidict<L, R> b{m.cbegin(), m.cend()};

    return b;
  }
  static void to_json(json &j, ::FlexFlow::bidict<L, R> const &b) {
    CHECK_IS_JSON_SERIALIZABLE(L);
    CHECK_IS_JSON_SERIALIZABLE(R);

    j = b.as_map();
  }
};

} // namespace nlohmann

namespace rc {

template <typename L, typename R>
struct Arbitrary<::FlexFlow::bidict<L, R>> {
  static Gen<::FlexFlow::bidict<L, R>> arbitrary() {
    return gen::map(
        gen::withSize([](int size) -> Gen<std::map<L, R>> {
          return gen::apply(
              [](std::vector<L> const &keys,
                 std::vector<R> const &values) -> std::map<L, R> {
                return ::FlexFlow::map_from_keys_and_values(keys, values);
              },
              gen::unique<std::vector<L>>(size, gen::arbitrary<L>()),
              gen::unique<std::vector<R>>(size, gen::arbitrary<R>()));
        }),
        [](std::map<L, R> const &m) {
          return ::FlexFlow::bidict<L, R>{m.cbegin(), m.cend()};
        });
  }
};

} // namespace rc

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::bidict<L, R>> {
  size_t operator()(::FlexFlow::bidict<L, R> const &b) const {
    return hash<std::map<L, R>>{}(b.as_map());
  }
};

} // namespace std

#endif

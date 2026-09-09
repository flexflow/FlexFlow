#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_UNORDERED_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_UNORDERED_BIDICT_H

#include "utils/check_fmtable.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/map_from_unordered.h"
#include "utils/containers/require_same.h"
#include "utils/containers/unordered_keys.h"
#include "utils/containers/unordered_map_from_keys_and_values.h"
#include "utils/containers/unordered_set_of.h"

#include "utils/containers/values.h"
#include "utils/fmt/unordered_map.h"
#include "utils/hash/unordered_map.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include <cassert>
#include <nlohmann/json.hpp>
#include <optional>
#include <rapidcheck.h>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename R>
struct unordered_bidict {
  unordered_bidict() : fwd_map{}, bwd_map{} {}

  unordered_bidict(std::initializer_list<std::pair<L, R>> init)
      : unordered_bidict(init.begin(), init.end()) {}

  template <typename InputIt>
  unordered_bidict(InputIt first, InputIt last) {
    for (auto it = first; it != last; it++) {
      this->equate_strict(it->first, it->second);
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

  void equate(L const &l, R const &r) {
    bool contains_l = this->contains_l(l);
    bool contains_r = this->contains_r(r);

    if (contains_l != contains_r || (contains_l && this->at_l(l) != r)) {
      this->erase_l(l);
      this->erase_r(r);
      contains_l = contains_r = false;
    }

    if (!contains_l) {
      ASSERT(!contains_r);
      fwd_map.insert({l, r});
      bwd_map.insert({r, l});
    }
  }

  void equate(std::pair<L, R> const &lr) {
    this->equate(lr.first, lr.second);
  }

  void equate_strict(L const &l, R const &r) {
    ASSERT(this->contains_l(l) == this->contains_r(r));

    if (this->contains_l(l)) {
      ASSERT(this->at_l(l) == r);
    } else {
      fwd_map.insert({l, r});
      bwd_map.insert({r, l});
    }
  }

  void equate_strict(std::pair<L, R> const &lr) {
    this->equate_strict(lr.first, lr.second);
  }

  bool operator==(unordered_bidict<L, R> const &other) const {
    return require_same((this->fwd_map == other.fwd_map),
                        (this->bwd_map == other.bwd_map));
  }

  bool operator!=(unordered_bidict<L, R> const &other) const {
    return require_same((this->fwd_map != other.fwd_map),
                        (this->bwd_map != other.bwd_map));
  }

  R const &at_l(L const &l) const {
    ASSERT(contains_key(this->fwd_map, l));
    return fwd_map.at(l);
  }

  L const &at_r(R const &r) const {
    ASSERT(contains_key(this->bwd_map, r));
    return bwd_map.at(r);
  }

  std::unordered_set<L> left_values() const {
    return unordered_keys(this->fwd_map);
  }

  std::unordered_set<R> right_values() const {
    return unordered_keys(this->bwd_map);
  }

  std::size_t size() const {
    assert(fwd_map.size() == bwd_map.size());
    return fwd_map.size();
  }

  bool empty() const {
    return this->size() == 0;
  }

  using const_iterator = typename std::unordered_map<L, R>::const_iterator;
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

  unordered_bidict<R, L> reversed() const {
    return unordered_bidict<R, L>(bwd_map, fwd_map);
  }

  operator std::unordered_map<L, R> const &() const {
    return this->fwd_map;
  }

  operator std::map<L, R>() const {
    return map_from_unordered(this->fwd_map);
  }

  std::unordered_map<L, R> const &as_unordered_map() const {
    return this->fwd_map;
  }

  std::map<L, R> as_map() const {
    return map_from_unordered(this->fwd_map);
  }

  std::unordered_map<L, R> const &l_to_r() const {
    return this->fwd_map;
  }

  std::unordered_map<R, L> const &r_to_l() const {
    return this->bwd_map;
  }

  unordered_bidict(std::unordered_map<L, R> const &fwd_map,
                   std::unordered_map<R, L> const &bwd_map)
      : fwd_map(fwd_map), bwd_map(bwd_map) {
    this->check_invariants();
  }

  // note: std::unordered_map provides no relational operators, so ordering
  // comparisons are performed on the ordered projection of the forward map
  bool operator<(unordered_bidict<L, R> const &other) const {
    return this->as_map() < other.as_map();
  }

  bool operator<=(unordered_bidict<L, R> const &other) const {
    return this->as_map() <= other.as_map();
  }

  bool operator>(unordered_bidict<L, R> const &other) const {
    return this->as_map() > other.as_map();
  }

  bool operator>=(unordered_bidict<L, R> const &other) const {
    return this->as_map() >= other.as_map();
  }

private:
  void check_invariants() const {
    std::unordered_set<L> fwd_l_vals = unordered_keys(this->fwd_map);
    std::unordered_set<L> bwd_l_vals = unordered_set_of(values(this->bwd_map));

    std::unordered_set<R> bwd_r_vals = unordered_keys(this->bwd_map);
    std::unordered_set<R> fwd_r_vals = unordered_set_of(values(this->fwd_map));

    ASSERT(fwd_l_vals == bwd_l_vals);
    ASSERT(fwd_r_vals == bwd_r_vals);

    for (L const &l : fwd_l_vals) {
      ASSERT(bwd_map.at(fwd_map.at(l)) == l);
    }
  }

  friend struct unordered_bidict<R, L>;

  std::unordered_map<L, R> fwd_map;
  std::unordered_map<R, L> bwd_map;
};

template <typename L, typename R>
std::unordered_map<L, R> format_as(unordered_bidict<L, R> const &b) {
  return b.as_unordered_map();
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, unordered_bidict<L, R> const &b) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return s << fmt::to_string(b);
}

} // namespace FlexFlow

namespace nlohmann {

template <typename L, typename R>
struct adl_serializer<::FlexFlow::unordered_bidict<L, R>> {
  static ::FlexFlow::unordered_bidict<L, R> from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(L);
    CHECK_IS_JSON_DESERIALIZABLE(R);

    std::unordered_map<L, R> m = j;

    ::FlexFlow::unordered_bidict<L, R> b{m.cbegin(), m.cend()};

    return b;
  }
  static void to_json(json &j, ::FlexFlow::unordered_bidict<L, R> const &b) {
    CHECK_IS_JSON_SERIALIZABLE(L);
    CHECK_IS_JSON_SERIALIZABLE(R);

    // note: serialize via the ordered projection so that the emitted json
    // has a deterministic element order (std::unordered_map does not define
    // iteration order)
    j = b.as_map();
  }
};

} // namespace nlohmann

namespace rc {

template <typename L, typename R>
struct Arbitrary<::FlexFlow::unordered_bidict<L, R>> {
  static Gen<::FlexFlow::unordered_bidict<L, R>> arbitrary() {
    return gen::map(
        gen::withSize([](int size) -> Gen<std::unordered_map<L, R>> {
          return gen::apply(
              [](std::vector<L> const &keys,
                 std::vector<R> const &values) -> std::unordered_map<L, R> {
                return ::FlexFlow::unordered_map_from_keys_and_values(keys,
                                                                      values);
              },
              gen::unique<std::vector<L>>(size, gen::arbitrary<L>()),
              gen::unique<std::vector<R>>(size, gen::arbitrary<R>()));
        }),
        [](std::unordered_map<L, R> const &m) {
          return ::FlexFlow::unordered_bidict<L, R>{m.cbegin(), m.cend()};
        });
  }
};

} // namespace rc

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::unordered_bidict<L, R>> {
  size_t operator()(::FlexFlow::unordered_bidict<L, R> const &b) const {
    return hash<std::unordered_map<L, R>>{}(b.as_unordered_map());
  }
};

} // namespace std

#endif

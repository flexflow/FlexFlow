#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_INL
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_INL

#include "containers.decl.h"
#include "required_core.h"
#include "type_traits_core.h"
#include "utils/bidict/bidict.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/contains.h"
#include "utils/containers/extend.h"
#include "utils/containers/extend_vector.h"
#include "utils/containers/filter.h"
#include "utils/containers/intersection.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_transform.h"
#include "utils/exception.h"
#include "utils/hash/pair.h"
#include "utils/optional.h"
#include "utils/type_traits.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename Container, typename Element>
Element sum(Container const &container) {
  Element result = 0;
  for (Element const &element : container) {
    result += element;
  }
  return result;
}

template <typename Container, typename ConditionF, typename Element>
Element sum_where(Container const &container, ConditionF const &condition) {
  Element result = 0;
  for (Element const &element : container) {
    if (condition(element)) {
      result += element;
    }
  }
  return result;
}

template <typename Container, typename ConditionF, typename Element>
Element product_where(Container const &container, ConditionF const &condition) {
  Element result = 1;
  for (Element const &element : container) {
    if (condition(element)) {
      result *= element;
    }
  }
  return result;
}

template <typename K, typename V>
bool contains_l(bidict<K, V> const &m, K const &k) {
  return m.contains_l(k);
}

template <typename K, typename V>
bool contains_r(bidict<K, V> const &m, V const &v) {
  return m.contains_r(v);
}

template <typename K, typename V>
bool is_submap(std::unordered_map<K, V> const &m,
               std::unordered_map<K, V> const &sub) {
  return restrict_keys(m, keys(sub)) == sub;
}

template <typename Container, typename Element>
std::optional<std::size_t> index_of(Container const &c, Element const &e) {
  auto it = std::find(c.cbegin(), c.cend(), e);
  if (it == c.cend()) {
    return std::nullopt;
  } else {
    return std::distance(c.cbegin(), it);
  }
}

template <typename K, typename V>
std::unordered_map<K, V> merge_maps(std::unordered_map<K, V> const &lhs,
                                    std::unordered_map<K, V> const &rhs) {
  assert(are_disjoint(keys(lhs), keys(rhs)));

  std::unordered_map<K, V> result;
  for (auto const &kv : lhs) {
    result.insert(kv);
  }
  for (auto const &kv : rhs) {
    result.insert(kv);
  }

  return result;
}

template <typename K, typename V>
bidict<K, V> merge_maps(bidict<K, V> const &lhs, bidict<K, V> const &rhs) {
  assert(are_disjoint(keys(lhs), keys(rhs)));

  bidict<K, V> result;
  for (auto const &kv : lhs) {
    result.equate(kv.first, kv.second);
  }
  for (auto const &kv : rhs) {
    result.equate(kv.first, kv.second);
  }

  return result;
}

template <typename K, typename V>
std::function<V(K const &)> lookup_in(std::unordered_map<K, V> const &m) {
  return [&m](K const &k) -> V { return m.at(k); };
}

template <typename L, typename R>
std::function<R(L const &)> lookup_in_l(bidict<L, R> const &m) {
  return [&m](L const &l) -> R { return m.at_l(l); };
}

template <typename L, typename R>
std::function<L(R const &)> lookup_in_r(bidict<L, R> const &m) {
  return [&m](R const &r) -> L { return m.at_r(r); };
}

template <typename T>
bool is_superseteq_of(std::unordered_set<T> const &l,
                      std::unordered_set<T> const &r) {
  return is_subseteq_of<T>(r, l);
}

template <typename Container, typename Function>
std::optional<bool> optional_all_of(Container const &container,
                                    Function const &func) {
  for (auto const &element : container) {
    std::optional<bool> condition = func(element);
    if (!condition.has_value()) {
      return std::nullopt;
    }

    if (!condition.value()) {
      return false;
    }
  }
  return true;
}

template <typename C>
bool are_all_same(C const &c) {
  auto const &first = *c.cbegin();
  for (auto const &v : c) {
    if (v != first) {
      return false;
    }
  }
  return true;
}

template <typename In, typename F, typename Out>
std::vector<Out> flatmap(std::vector<In> const &v, F const &f) {
  std::vector<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename In, typename F, typename Out>
std::unordered_set<Out> flatmap(std::unordered_set<In> const &v, F const &f) {
  std::unordered_set<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename Out, typename In>
std::unordered_set<Out> flatmap_v2(std::unordered_set<In> const &v,
                                   std::unordered_set<Out> (*f)(In const &)) {
  std::unordered_set<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename T, typename F>
std::function<bool(T const &, T const &)> compare_by(F const &f) {
  return [=](T const &lhs, T const &rhs) { return f(lhs) < f(rhs); };
}

template <typename C>
typename C::value_type maximum(C const &v) {
  return *std::max_element(v.begin(), v.end());
}

template <typename T>
std::vector<T> value_all(std::vector<std::optional<T>> const &v) {
  return transform(v, [](std::optional<T> const &element) {
    return unwrap(element, [] {
      throw mk_runtime_error(
          "Encountered element without value in call to value_all");
    });
  });
}

template <typename T>
std::unordered_set<T> value_all(std::unordered_set<std::optional<T>> const &v) {
  return transform(v, [](std::optional<T> const &element) {
    return unwrap(element, [] {
      throw mk_runtime_error(
          "Encountered element without value in call to value_all");
    });
  });
}

template <typename C>
struct reversed_container_t {
  reversed_container_t() = delete;
  reversed_container_t(C const &c) : container(c) {}

  reversed_container_t(reversed_container_t const &) = delete;
  reversed_container_t(reversed_container_t &&) = delete;
  reversed_container_t &operator=(reversed_container_t const &) = delete;
  reversed_container_t &operator=(reversed_container_t &&) = delete;

  using iterator = typename C::reverse_iterator;
  using const_iterator = typename C::const_reverse_iterator;
  using reverse_iterator = typename C::iterator;
  using const_reverse_iterator = typename C::const_iterator;
  using value_type = typename C::value_type;
  using pointer = typename C::pointer;
  using const_pointer = typename C::const_pointer;
  using reference = typename C::reference;
  using const_reference = typename C::const_reference;

  iterator begin() {
    return this->container.rend();
  }

  iterator end() {
    return this->container.rbegin();
  }

  const_iterator cbegin() const {
    return this->container.crend();
  }

  const_iterator cend() const {
    return this->container.crbegin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator end() const {
    return this->cend();
  }

  reverse_iterator rbegin() {
    return this->container.begin();
  }

  reverse_iterator rend() {
    return this->container.end();
  }

  const_reverse_iterator crbegin() const {
    return this->container.cbegin();
  }

  const_reverse_iterator crend() const {
    return this->container.cend();
  }

  const_reverse_iterator rbegin() const {
    return this->crbegin();
  }

  const_reverse_iterator rend() const {
    return this->crend();
  }

private:
  C const &container;
};

template <typename C>
reversed_container_t<C> reversed_container(C const &c) {
  return reversed_container_t<C>(c);
}

} // namespace FlexFlow

#endif

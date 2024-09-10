#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_DECL_H

#include "utils/bidict/bidict.h"
#include "utils/containers/get_element_type.h"
#include "utils/required_core.h"
#include "utils/type_traits_core.h"
#include <optional>
#include <string>
#include <vector>

namespace FlexFlow {

template <typename Container, typename Element = typename Container::value_type>
Element sum(Container const &container);

template <typename Container,
          typename ConditionF,
          typename Element = typename Container::value_type>
Element sum_where(Container const &container, ConditionF const &condition);

template <typename Container,
          typename ConditionF,
          typename Element = typename Container::value_type>
Element product_where(Container const &container, ConditionF const &condition);

template <typename K, typename V>
bool contains_l(bidict<K, V> const &m, K const &k);

template <typename K, typename V>
bool contains_r(bidict<K, V> const &m, V const &v);

template <typename Container, typename Element>
std::optional<std::size_t> index_of(Container const &c, Element const &e);

template <typename K, typename V>
std::unordered_map<K, V> restrict_keys(std::unordered_map<K, V> const &m,
                                       std::unordered_set<K> const &mask);

template <typename K, typename V>
std::unordered_map<K, V> merge_maps(std::unordered_map<K, V> const &lhs,
                                    std::unordered_map<K, V> const &rhs);

template <typename E>
std::optional<E> at_idx(std::vector<E> const &v, size_t idx);

template <typename K, typename V>
std::function<V(K const &)> lookup_in(std::unordered_map<K, V> const &m);

template <typename L, typename R>
std::function<R(L const &)> lookup_in_l(bidict<L, R> const &m);

template <typename L, typename R>
std::function<L(R const &)> lookup_in_r(bidict<L, R> const &m);

template <typename T>
bool is_superseteq_of(std::unordered_set<T> const &l,
                      std::unordered_set<T> const &r);

template <typename Container, typename Function>
std::optional<bool> optional_all_of(Container const &, Function const &);

template <typename C>
bool are_all_same(C const &c);

template <typename In, typename F, typename Out>
std::vector<Out> flatmap(std::vector<In> const &v, F const &f);

template <typename In, typename F, typename Out>
std::unordered_set<Out> flatmap(std::unordered_set<In> const &v, F const &f);
template <typename Out, typename In>
std::unordered_set<Out> flatmap_v2(std::unordered_set<In> const &v,
                                   std::unordered_set<Out> (*f)(In const &));

template <typename T, typename F>
std::function<bool(T const &, T const &)> compare_by(F const &f);

template <typename C>
typename C::value_type maximum(C const &v);

template <typename T>
std::vector<T> value_all(std::vector<std::optional<T>> const &v);

template <typename T>
std::unordered_set<T> value_all(std::unordered_set<std::optional<T>> const &v);

template <typename C>
struct reversed_container_t;

template <typename C>
reversed_container_t<C> reversed_container(C const &c);

} // namespace FlexFlow

#endif

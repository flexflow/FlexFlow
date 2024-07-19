#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_DECL_H

#include "utils/bidict/bidict.h"
#include "utils/required_core.h"
#include "utils/type_traits_core.h"
#include <optional>
#include <string>
#include <vector>
#include "utils/containers/sorted.h"

namespace FlexFlow {

template <typename C, typename Enable = void>
struct get_element_type;

template <typename T>
using get_element_type_t = typename get_element_type<T>::type;

template <typename Container>
typename Container::const_iterator
    find(Container const &c, typename Container::value_type const &e);

template <typename Container, typename Element = typename Container::value_type>
Element sum(Container const &container);

template <typename Container,
          typename ConditionF,
          typename Element = typename Container::value_type>
Element sum_where(Container const &container, ConditionF const &condition);

template <typename Container, typename Element = typename Container::value_type>
Element product(Container const &container);

template <typename Container,
          typename ConditionF,
          typename Element = typename Container::value_type>
Element product_where(Container const &container, ConditionF const &condition);

template <typename It>
typename It::value_type product(It begin, It end);

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e);

template <typename K, typename V>
bool contains_l(bidict<K, V> const &m, K const &k);

template <typename K, typename V>
bool contains_r(bidict<K, V> const &m, V const &v);

template <typename K,
          typename V,
          typename F,
          typename K2 = decltype(std::declval<F>()(std::declval<K>()))>
std::unordered_map<K2, V> map_keys(std::unordered_map<K, V> const &m,
                                   F const &f);

template <typename K, typename V, typename F>
std::unordered_map<K, V> filter_keys(std::unordered_map<K, V> const &m,
                                     F const &f);

template <typename K,
          typename V,
          typename F,
          typename V2 = decltype(std::declval<F>()(std::declval<V>()))>
std::unordered_map<K, V2> map_values(std::unordered_map<K, V> const &m,
                                     F const &f);

template <typename K, typename V, typename F>
std::unordered_map<K, V> filter_values(std::unordered_map<K, V> const &m,
                                       F const &f);

template <typename C>
std::unordered_set<typename C::key_type> keys(C const &c);

template <typename C>
std::vector<typename C::mapped_type> values(C const &c);

template <typename C>
std::unordered_set<std::pair<typename C::key_type, typename C::mapped_type>>
    items(C const &c);

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> unique(C const &c);

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> without_order(C const &c);

template <typename Container, typename Element>
std::optional<std::size_t> index_of(Container const &c, Element const &e);

template <typename T>
std::unordered_set<T> intersection(std::unordered_set<T> const &l,
                                   std::unordered_set<T> const &r);

template <typename C, typename T = typename C::value_type>
std::optional<T> intersection(C const &c);

template <typename T>
bool are_disjoint(std::unordered_set<T> const &l,
                  std::unordered_set<T> const &r);

template <typename K, typename V>
std::unordered_map<K, V> restrict_keys(std::unordered_map<K, V> const &m,
                                       std::unordered_set<K> const &mask);

template <typename K, typename V>
std::unordered_map<K, V> merge_maps(std::unordered_map<K, V> const &lhs,
                                    std::unordered_map<K, V> const &rhs);

template <typename K, typename V>
bidict<K, V> merge_maps(bidict<K, V> const &lhs, bidict<K, V> const &rhs);

template <typename F,
          typename C,
          typename K = get_element_type_t<C>,
          typename V = std::invoke_result_t<F, K>>
std::unordered_map<K, V> generate_map(C const &c, F const &f);

template <typename F,
          typename C,
          typename K = get_element_type_t<C>,
          typename V = std::invoke_result_t<F, K>>
bidict<K, V> generate_bidict(C const &c, F const &f);

template <typename E>
std::optional<E> at_idx(std::vector<E> const &v, size_t idx);

template <typename K, typename V>
std::function<V(K const &)> lookup_in(std::unordered_map<K, V> const &m);

template <typename L, typename R>
std::function<R(L const &)> lookup_in_l(bidict<L, R> const &m);

template <typename L, typename R>
std::function<L(R const &)> lookup_in_r(bidict<L, R> const &m);

template <typename T>
std::unordered_set<T> set_union(std::unordered_set<T> const &l,
                                std::unordered_set<T> const &r);

template <typename T>
std::unordered_set<T> set_difference(std::unordered_set<T> const &l,
                                     std::unordered_set<T> const &r);

template <typename C, typename T = typename C::value_type::value_type>
std::unordered_set<T> set_union(C const &sets);

template <typename T>
bool is_subseteq_of(std::unordered_set<T> const &l,
                    std::unordered_set<T> const &r);

template <typename T>
bool is_supserseteq_of(std::unordered_set<T> const &l,
                       std::unordered_set<T> const &r);

template <typename S, typename D>
std::unordered_set<D>
    map_over_unordered_set(std::function<D(S const &)> const &f,
                           std::unordered_set<S> const &input);

template <typename C>
std::optional<typename C::value_type> maybe_get_only(C const &c);

template <typename C>
typename C::value_type get_only(C const &c);

template <typename T>
T get_first(std::unordered_set<T> const &s);

template <typename T, typename C>
void extend(std::vector<T> &lhs, C const &rhs);

template <typename T, typename C>
void extend(std::unordered_set<T> &lhs, C const &rhs);

template <typename C, typename E = typename C::value_type>
void extend(C &lhs, std::optional<E> const &e);

template <typename C, typename F>
bool all_of(C const &c, F const &f);

template <typename Container, typename Function>
std::optional<bool> optional_all_of(Container const &, Function const &);

template <typename C, typename F>
int count(C const &c, F const &f);

template <typename C>
bool are_all_same(C const &c);

template <typename C, typename E = typename C::value_type>
std::vector<E> as_vector(C const &c);

template <typename F,
          typename In,
          typename Out = decltype(std::declval<F>()(std::declval<In>()))>
std::vector<Out> transform(std::vector<In> const &v, F const &f);

template <typename F, typename C>
auto transform(req<C> const &c, F const &f)
    -> decltype(transform(std::declval<C>(), std::declval<F>()));

template <typename F,
          typename In,
          typename Out = decltype(std::declval<F>()(std::declval<In>()))>
std::unordered_set<Out> transform(std::unordered_set<In> const &v, F const &f);

template <typename F>
std::string transform(std::string const &s, F const &f);

template <typename F, typename Out = std::invoke_result_t<F>>
std::vector<Out> repeat(int n, F const &f);

template <typename T>
bidict<size_t, T> enumerate(std::unordered_set<T> const &c);

std::vector<size_t> count(size_t n);

template <typename In,
          typename F,
          typename Out = typename decltype(std::declval<F>()(
              std::declval<In>()))::value_type>
std::vector<Out> flatmap(std::vector<In> const &v, F const &f);

template <typename In,
          typename F,
          typename Out = get_element_type_t<std::invoke_result_t<F, In>>>
std::unordered_set<Out> flatmap(std::unordered_set<In> const &v, F const &f);

template <typename Out, typename In>
std::unordered_set<Out> flatmap_v2(std::unordered_set<In> const &v,
                                   std::unordered_set<Out> (*f)(In const &));

template <typename T, typename F>
std::function<bool(T const &, T const &)> compare_by(F const &f);

template <typename T>
std::pair<std::vector<T>, std::vector<T>> vector_split(std::vector<T> const &v,
                                                       std::size_t idx);

template <typename C>
typename C::value_type maximum(C const &v);

template <typename T>
T reversed(T const &t);

template <typename T>
std::vector<T> value_all(std::vector<std::optional<T>> const &v);

template <typename T>
std::unordered_set<T> value_all(std::unordered_set<std::optional<T>> const &v);

template <typename T>
std::vector<T> subvec(std::vector<T> const &v,
                      std::optional<int> const &maybe_start,
                      std::optional<int> const &maybe_end);

template <typename C>
struct reversed_container_t;

template <typename C>
reversed_container_t<C> reversed_container(C const &c);

} // namespace FlexFlow

#endif

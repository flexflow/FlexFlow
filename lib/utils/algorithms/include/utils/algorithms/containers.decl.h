#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_DECL_H

#include "utils/bidict.h"
#include "utils/invoke.h"
#include "utils/optional.decl.h"
#include "utils/required_core.h"
#include "utils/sequence.decl.h"
#include "utils/type_traits.decl.h"
#include "utils/type_traits_core.h"
#include <string>
#include <vector>

namespace FlexFlow {

template <typename C, typename Enable = void>
struct get_element_type;

template <typename T>
using get_element_type_t = typename get_element_type<T>::type;

template <typename InputIt, typename F>
std::string join_strings(InputIt first,
                         InputIt last,
                         std::string const &delimiter,
                         F const &f);

template <typename InputIt>
std::string
    join_strings(InputIt first, InputIt last, std::string const &delimiter);

template <typename Container>
std::string join_strings(Container const &c, std::string const &delimiter);

template <typename Container, typename F>
std::string
    join_strings(Container const &c, std::string const &delimiter, F const &f);

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

template <typename C>
bool contains_key(C const &m, typename C::key_type const &k);

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

template <typename K,
          typename V,
          typename F,
          typename K2 = decltype(std::declval<F>()(std::declval<K>()))>
bidict<K2, V> map_keys(bidict<K, V> const &m, F const &f);

template <typename K, typename V, typename F>
std::unordered_map<K, V> filter_keys(std::unordered_map<K, V> const &m,
                                     F const &f);

template <typename K, typename V, typename F>
bidict<K, V> filter_values(bidict<K, V> const &m, F const &f);

template <typename K,
          typename V,
          typename F,
          typename V2 = decltype(std::declval<F>()(std::declval<V>()))>
std::unordered_map<K, V2> map_values(std::unordered_map<K, V> const &m,
                                     F const &f);

template <typename K,
          typename V,
          typename F,
          typename V2 = decltype(std::declval<F>()(std::declval<V>()))>
bidict<K, V2> map_values(bidict<K, V> const &m, F const &f);

template <typename K, typename V, typename F>
std::unordered_map<K, V> filter_values(std::unordered_map<K, V> const &m,
                                       F const &f);

template <typename C>
std::unordered_set<typename C::key_type> keys(C const &c);

template <typename C>
std::vector<typename C::mapped_type> values(C const &c);

template <typename C>
std::unordered_set<std::pair<typename C::key_type, typename C::value_type>>
    items(C const &c);

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> unique(C const &c);

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> without_order(C const &c);

template <typename Container, typename Element>
optional<std::size_t> index_of(Container const &c, Element const &e);

template <typename T>
std::unordered_set<T> intersection(std::unordered_set<T> const &l,
                                   std::unordered_set<T> const &r);

template <typename C, typename T = typename C::value_type>
optional<T> intersection(C const &c);

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
          typename V = invoke_result_t<F, K>>
std::unordered_map<K, V> generate_map(C const &c, F const &f);

template <typename F,
          typename C,
          typename K = get_element_type_t<C>,
          typename V = invoke_result_t<F, K>>
bidict<K, V> generate_bidict(C const &c, F const &f);

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
optional<typename C::value_type> maybe_get_only(C const &c);

template <typename C>
typename C::value_type get_only(C const &c);

template <typename T>
T get_first(std::unordered_set<T> const &s);

template <typename T, typename C>
void extend(std::vector<T> &lhs, C const &rhs);

template <typename T, typename C>
void extend(std::unordered_set<T> &lhs, C const &rhs);

template <typename C, typename E = typename C::value_type>
void extend(C &lhs, optional<E> const &e);

template <typename C, typename F>
bool all_of(C const &c, F const &f);

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

template <typename F, typename... Ts>
auto transform(std::tuple<Ts...> const &tup, F const &f);

template <typename F, typename... Ts>
void for_each(std::tuple<Ts...> const &tup, F const &f);

template <typename Head, typename... Rest>
enable_if_t<types_are_all_same_v<Head, Rest...>, std::vector<Head>>
    to_vector(std::tuple<Head, Rest...> const &tup);

template <typename F, typename C>
auto transform(req<C> const &c, F const &f)
    -> decltype(transform(std::declval<C>(), std::declval<F>()));

template <typename F,
          typename In,
          typename Out = decltype(std::declval<F>()(std::declval<In>()))>
std::vector<Out> vector_transform(F const &f, std::vector<In> const &v);

template <typename F,
          typename In,
          typename Out = decltype(std::declval<F>()(std::declval<In>()))>
std::unordered_set<Out> transform(std::unordered_set<In> const &v, F const &f);

template <typename F>
std::string transform(std::string const &s, F const &f);

template <typename F, typename Out = invoke_result_t<F>>
std::vector<Out> repeat(int n, F const &f);

template <typename T>
bidict<size_t, T> enumerate(std::unordered_set<T> const &c);

std::vector<size_t> count(size_t n);

template <typename In,
          typename F,
          typename Out = get_element_type_t<std::invoke_result_t<F, In>>>
std::vector<Out> flatmap(std::vector<In> const &v, F const &f);

template <typename F,
          typename = std::enable_if_t<
              std::is_same_v<std::decay_t<std::invoke_result_t<F, char>>,
                             std::string>>>
std::string flatmap(std::string const &v, F const &f);

template <typename In,
          typename F,
          typename Out = get_element_type_t<invoke_result_t<F, In>>>
std::unordered_set<Out> flatmap(std::unordered_set<In> const &v, F const &f);

template <typename Out, typename In>
std::unordered_set<Out> flatmap_v2(std::unordered_set<In> const &v,
                                   std::unordered_set<Out> (*f)(In const &));

template <typename C, typename F, typename Elem = typename C::value_type>
void inplace_sorted_by(C &c, F const &f);

template <typename C, typename F>
std::vector<sorted_elem_type_t<C>> sorted_by(C const &c, F const &f);

template <typename C>
std::vector<sorted_elem_type_t<C>> sorted(C const &c);

template <typename T, typename F>
std::function<bool(T const &, T const &)> compare_by(F const &f);

template <typename C, typename F>
C filter(C const &v, F const &f);

template <typename T, typename F>
std::unordered_set<T> filter(std::unordered_set<T> const &v, F const &f);

template <typename C, typename F, typename Elem = typename C::value_type>
void inplace_filter(C &v, F const &f);

template <typename T>
std::pair<std::vector<T>, std::vector<T>> vector_split(std::vector<T> const &v,
                                                       std::size_t idx);

template <typename C>
typename C::value_type maximum(C const &v);

template <typename T>
T reversed(T const &t);

template <typename T>
std::vector<T> value_all(std::vector<optional<T>> const &v);

template <typename T>
std::vector<T> subvec(std::vector<T> const &v,
                      optional<int> const &maybe_start,
                      optional<int> const &maybe_end);

template <typename C>
struct reversed_container_t;

template <typename C>
reversed_container_t<C> reversed_container(C const &c);

} // namespace FlexFlow

#endif

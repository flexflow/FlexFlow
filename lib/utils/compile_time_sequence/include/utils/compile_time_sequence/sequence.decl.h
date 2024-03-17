/* #ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_DECL_H */
/* #define _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_DECL_H */

/* #include "utils/optional.decl.h" */
/* #include <tuple> */

/* namespace FlexFlow { */

/* template <int... S> */
/* struct seq; */

/* template <typename Seq> */
/* struct seq_head; */
/* template <typename Seq> */
/* inline constexpr int seq_head_v = seq_head<Seq>::value; */

/* template <typename Seq> */
/* struct seq_tail; */
/* template <typename Seq> */
/* using seq_tail_t = typename seq_tail<Seq>::type; */

/* template <int X, typename Seq> */
/* struct seq_prepend; */
/* template <int X, typename Seq> */
/* using seq_prepend_t = typename seq_prepend<X, Seq>::type; */

/* template <typename Seq, int ToAppend> */
/* struct seq_append; */
/* template <typename Seq, int ToAppend> */
/* using seq_append_t = typename seq_append<Seq, ToAppend>::type; */

/* template <int n> */
/* struct seq_count; */
/* template <int n> */
/* using seq_count_t = typename seq_count<n>::type; */

/* template <typename... Args> */
/* struct seq_enumerate_args; */
/* template <typename... Args> */
/* using seq_enumerate_args_t = typename seq_enumerate_args<Args...>::type; */

/* template <typename T> */
/* struct seq_enumerate_tuple; */
/* template <typename T> */
/* using seq_enumerate_tuple_t = typename seq_enumerate_tuple<T>::type; */

/* template <typename F, typename Seq> */
/* struct seq_transform_type; */
/* template <typename F, typename Seq> */
/* using seq_transform_type_t = typename seq_transform_type<F, Seq>::type; */

/* template <typename F, int... S> */
/* auto seq_transform(F const &f, seq<S...> const &) */
/*     -> seq_transform_type_t<F, seq<S...>>; */

/* template <typename F, typename... Ts> */
/* void seq_for_each(F const &f, std::tuple<Ts...> const &); */

/* template <typename F, int... S> */
/* auto seq_select(F const &f, int i, seq<S...> const &s) */
/*     -> decltype(f(std::declval<std::integral_constant<int, 0>>())); */

/* template <typename F, int... S> */
/* auto seq_get(F const &f, int i, seq<S...> const &s) */
/*     -> decltype(f(std::declval<std::integral_constant<int, 0>>())); */

/* } // namespace FlexFlow */

/* #endif */

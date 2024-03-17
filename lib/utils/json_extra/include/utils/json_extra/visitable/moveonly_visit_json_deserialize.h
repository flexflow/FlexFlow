#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_MOVEONLY_VISIT_JSON_DESERIALIZE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_MOVEONLY_VISIT_JSON_DESERIALIZE_H

namespace FlexFlow {

template <int idx, typename T>
typename std::enable_if<(idx >= std::tuple_size<visit_as_tuple_t<T>>::value),
                        std::tuple<>>::type
    tuple_from_json_impl(json const &j) {
  return std::tuple<>{};
}

template <int idx, typename T, typename Enable = void>
struct TupleFromJson {
  tuple_tail_t<idx, visit_as_tuple_t<T>> operator()(json const &j) {
    using FieldT = visit_struct::type_at<idx, T>;

    FieldT field =
        j.at(visit_struct::get_name<idx, T>()).template get<FieldT>();

    return std::tuple_cat(std::tuple<FieldT>(field),
                          TupleFromJson<(idx + 1), T>{}(j));
  }
};

template <int idx, typename T>
struct TupleFromJson<
    idx,
    T,
    typename std::enable_if<(
        idx > std::tuple_size<visit_as_tuple_t<T>>::value)>::type> {
  std::tuple<> operator()(json const &j) {
    return {};
  }
};

template <typename T>
visit_as_tuple_t<T> tuple_from_json(json const &j) {
  return TupleFromJson<0, T>{}(j);
}

template <typename T>
T moveonly_visit_json_deserialize(json const &j) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(!std::is_default_constructible<T>::value, "");
  static_assert(pretty_elements_satisfy<is_json_deserializable, T>::value,
                "Elements must be deserializable");

  return visitable_from_tuple<T>(tuple_from_json<T>(j));
}

} // namespace FlexFlow

#endif

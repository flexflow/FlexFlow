#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_JSON_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_JSON_H

#include "utils/json_core.h"
#include "utils/optional.h"
#include "utils/sequence.h"
#include "utils/type_traits.h"
#include "utils/variant.h"
#include "utils/visitable.h"
#include "utils/json/is_json_deserializable.h"
#include "utils/json/is_json_serializable.h"
#include "utils/json/is_jsonable.h"

namespace FlexFlow {

struct json_serialization_visitor {
  json_serialization_visitor() = delete;
  json_serialization_visitor(json &j) : j(j) {}

  json &j;

  template <typename T>
  void operator()(char const *field_name, T const &field_value) {
    j[field_name] = field_value;
  }
};

struct json_deserialization_visitor {
  json_deserialization_visitor() = delete;
  json_deserialization_visitor(json const &j) : j(j) {}

  json const &j;

  template <typename T>
  void operator()(char const *field_name, T &field_value) {
    j.at(field_name).get_to(field_value);
  }
};

static_assert(std::is_same<tuple_tail_t<0, std::tuple<int, float, bool>>,
                           std::tuple<int, float, bool>>::value,
              "");
static_assert(std::is_same<tuple_tail_t<3, std::tuple<int, float, bool>>,
                           std::tuple<>>::value,
              "");

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
void visit_json_serialize(json &j, T const &t) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_json_serializable, T>::value,
                "Elements must be deserializable");

  json_serialization_visitor vis(j);
  visit_struct::for_each(t, vis);
}

template <typename T>
void visit_json_deserialize(json const &j, T &t) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_json_deserializable, T>::value,
                "Elements must be deserializable");

  json_deserialization_visitor vis(j);
  visit_struct::for_each(t, vis);
}

template <typename T>
T moveonly_visit_json_deserialize(json const &j) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(!std::is_default_constructible<T>::value, "");
  static_assert(elements_satisfy<is_json_deserializable, T>::value,
                "Elements must be deserializable");

  return visitable_from_tuple<T>(tuple_from_json<T>(j));
}


} // namespace FlexFlow

namespace nlohmann {

template <typename T>
struct adl_serializer<
    T,
    typename std::enable_if<::FlexFlow::conjunction<
        ::FlexFlow::is_visitable<T>,
        ::FlexFlow::elements_satisfy<::FlexFlow::is_json_serializable, T>,
        std::is_default_constructible<T>>::value>::type> {
  static void to_json(json &j, T const &t) {
    ::FlexFlow::visit_json_serialize(j, t);
  }

  static void from_json(json const &j, T &t) {
    ::FlexFlow::visit_json_deserialize(j, t);
  }
};

template <typename T>
struct adl_serializer<
    T,
    typename std::enable_if<::FlexFlow::conjunction<
        ::FlexFlow::is_visitable<T>,
        ::FlexFlow::elements_satisfy<::FlexFlow::is_json_serializable, T>,
        ::FlexFlow::negation<std::is_default_constructible<T>>,
        std::is_move_constructible<T>>::value>::type> {
  static void to_json(json &j, T const &t) {
    ::FlexFlow::visit_json_serialize(j, t);
  }

  static T from_json(json const &j) {
    return ::FlexFlow::moveonly_visit_json_deserialize<T>(j);
  }
};

} // namespace nlohmann

#endif

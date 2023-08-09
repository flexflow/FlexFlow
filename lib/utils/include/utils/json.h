#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_JSON_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_JSON_H

#include "utils/json_core.h"
#include "utils/optional.h"
#include "utils/sequence.h"
#include "utils/type_traits.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_json_serializable : std::false_type {};

template <typename T>
struct is_json_serializable<
    T,
    void_t<decltype(std::declval<json>() = std::declval<T>())>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_json_deserializable : std::false_type {};

template <typename T>
struct is_json_deserializable<T,
                              void_t<decltype(std::declval<json>().get<T>())>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_jsonable
    : conjunction<is_json_serializable<T>, is_json_deserializable<T>> {};

#define CHECK_IS_JSONABLE(TYPENAME)                                            \
  static_assert(is_json_serializable<TYPENAME>::value,                         \
                #TYPENAME " should be json serializeable");                    \
  static_assert(is_json_deserializable<TYPENAME>::value,                       \
                #TYPENAME " should be json deserializeable")

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

struct VariantToJsonFunctor {
  VariantToJsonFunctor(json &j) : j(j) {}

  json &j;

  template <typename T>
  void operator()(T const &t) {
    static_assert(is_jsonable<T>::value, "");

    j["type"] = get_name(t);
    j["value"] = t;
  }
};

template <typename... Args>
void variant_to_json(json &j, variant<Args...> const &v) {
  visit(::FlexFlow::VariantToJsonFunctor{j}, v.value);
}

template <typename Variant>
struct VariantFromJsonFunctor {
  VariantFromJsonFunctor(json const &j) : j(j) {}

  json const &j;

  template <int Idx>
  optional<Variant> operator()(std::integral_constant<int, Idx> const &) const {
    using Type = typename variant_alternative<Idx, Variant>::type;

    if (visit_struct::get_name<Type>()) {
      return j.at("value").get<Type>();
    }
  }
};

template <typename... Args>
variant<Args...> variant_from_json(json const &j) {
  ::FlexFlow::VariantFromJsonFunctor<::FlexFlow::variant<Args...>> func(j);
  auto result = seq_map(func, seq_enumerate_args_t<Args...>{});
  if (!result.has_value()) {
    throw ::FlexFlow::mk_runtime_error("Invalid type {} found in json",
                                       j.at("type").get<std::string>());
  }
  return result.value();
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

template <typename T>
struct adl_serializer<
    ::FlexFlow::optional<T>,
    typename std::enable_if<::FlexFlow::is_jsonable<T>::value>::type> {
  static void to_json(json &j, ::FlexFlow::optional<T> const &t) {
    if (t.has_value()) {
      to_json(j, t.value());
    } else {
      j = nullptr;
    }
  }

  static void from_json(json const &j, ::FlexFlow::optional<T> &t) {
    if (j == nullptr) {
      t = ::FlexFlow::nullopt;
    } else {
      t = j.get<T>();
    }
  }
};

template <typename... Args>
struct adl_serializer<::FlexFlow::variant<Args...>,
                      typename std::enable_if<::FlexFlow::elements_satisfy<
                          ::FlexFlow::is_json_serializable,
                          ::FlexFlow::variant<Args...>>::value>::type> {
  static void to_json(json &j, ::FlexFlow::variant<Args...> const &v) {
    return ::FlexFlow::variant_to_json(j, v);
  }

  static ::FlexFlow::variant<Args...> from_json(json const &j) {
    return ::FlexFlow::variant_from_json<Args...>(j);
  }
};

} // namespace nlohmann

#endif

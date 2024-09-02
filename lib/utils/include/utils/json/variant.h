#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_VARIANT_H

#include <nlohmann/json.hpp>
#include "utils/json/is_jsonable.h"

namespace FlexFlow {

struct VariantToJsonFunctor {
  VariantToJsonFunctor(nlohmann::json &j) : j(j) {}

  nlohmann::json &j;

  template <typename T>
  void operator()(T const &t) {
    static_assert(is_jsonable<T>::value, "");

    j = t;
  }
};

template <typename... Args>
void variant_to_json(json &j, std::variant<Args...> const &v) {
  json jval;
  visit(::FlexFlow::VariantToJsonFunctor{jval}, v);
  j["value"] = jval;
  j["index"] = v.index();
}

template <typename Variant, size_t Idx>
std::optional<Variant> variant_from_json_impl(json const &j) {
  using Type = typename std::variant_alternative<Idx, Variant>::type;

  if (j.at("index").get<size_t>() == Idx) {
    return j.at("value").get<Type>();
  }
  return std::nullopt;
}

template <typename Variant, size_t... Is>
std::optional<Variant> variant_from_json_impl(json const &j,
                                              std::index_sequence<Is...>) {
  // If there were no errors when parsing, all but one element of the array
  // will be nullopt. This is because each call to variant_from_json_impl will
  // have a unique index and exactly one of them will match the index in the
  // json object.
  std::array<std::optional<Variant>, sizeof...(Is)> results{
      variant_from_json_impl<Variant, Is>(j)...};
  for (std::optional<Variant> &maybe : results) {
    if (maybe) {
      return maybe.value();
    }
  }
  return std::nullopt;
}

template <typename... Args>
std::variant<Args...> variant_from_json(json const &j) {
  using Variant = std::variant<Args...>;
  std::optional<Variant> result = variant_from_json_impl<Variant>(
      j, std::make_index_sequence<sizeof...(Args)>());
  if (!result.has_value()) {
    throw ::FlexFlow::mk_runtime_error("Invalid type {} found in json",
                                       j.at("index").get<size_t>());
  }
  return result.value();
}


} // namespace FlexFlow

namespace nlohmann {

template <typename... Args>
struct adl_serializer<std::variant<Args...>,
                      typename std::enable_if<::FlexFlow::elements_satisfy<
                          ::FlexFlow::is_json_serializable,
                          std::variant<Args...>>::value>::type> {
  static void to_json(json &j, std::variant<Args...> const &v) {
    return ::FlexFlow::variant_to_json(j, v);
  }

  static std::variant<Args...> from_json(json const &j) {
    return ::FlexFlow::variant_from_json<Args...>(j);
  }
};

} // namespace nlohmann

#endif

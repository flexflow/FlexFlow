#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VARIANT_VARIANT_FROM_JSON_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VARIANT_VARIANT_FROM_JSON_H

#include "nlohmann/json.hpp"

namespace FlexFlow {

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
  ::FlexFlow::VariantFromJsonFunctor<std::variant<Args...>> func(j);
  auto result = seq_map(func, seq_enumerate_args_t<Args...>{});
  if (!result.has_value()) {
    throw ::FlexFlow::mk_runtime_error("Invalid type {} found in json",
                                       j.at("type").get<std::string>());
  }
  return result.value();
}

} // namespace FlexFlow

#endif

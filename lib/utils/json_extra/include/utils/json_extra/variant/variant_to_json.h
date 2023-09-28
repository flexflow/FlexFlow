#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VARIANT_VARIANT_TO_JSON_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VARIANT_VARIANT_TO_JSON_H

namespace FlexFlow {

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


} // namespace FlexFlow

#endif

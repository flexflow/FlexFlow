#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VARIANT_SERIALIZER_INSTANCE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VARIANT_SERIALIZER_INSTANCE_H

namespace nlohmann {

template <typename... Args>
struct adl_serializer<std::variant<Args...>,
                      typename std::enable_if<::FlexFlow::elements_satisfy<
                          ::FlexFlow::is_json_serializable,
                          std::variant<Args...>>::value>::type> {
  static void to_json(json &j, std::variant<Args...> const &v) {
    return std::variant_to_json(j, v);
  }

  static std::variant<Args...> from_json(json const &j) {
    return std::variant_from_json<Args...>(j);
  }
};

} // namespace FlexFlow

#endif

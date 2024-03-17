#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_OPTIONAL_SERIALIZER_INSTANCE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_OPTIONAL_SERIALIZER_INSTANCE_H

namespace nlohmann {

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

} // namespace nlohmann

#endif

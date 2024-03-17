#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_REQUIRED_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_REQUIRED_H

namespace nlohmann {

template <typename T>
struct adl_serializer<::FlexFlow::req<T>> {
  static ::FlexFlow::req<T> from_json(json const &j) {
    return {j.template get<T>()};
  }

  static void to_json(json &j, ::FlexFlow::req<T> const &t) {
    j = static_cast<T>(t);
  }
};

} // namespace nlohmann

#endif

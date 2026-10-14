#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_OPTIONAL_H

#include "utils/commutative_pair.h"
#include "utils/json/is_jsonable.h"
#include <nlohmann/json.hpp>
#include <optional>

namespace nlohmann {

template <typename TT>
struct adl_serializer<std::optional<TT>> {
  // typename std::enable_if<::FlexFlow::is_jsonable<T>::value>::type> {

  template <typename T = TT>
  static std::enable_if_t<::FlexFlow::is_json_serializable_v<T>, void>
      to_json(json &j, std::optional<T> const &t) {
    if (t.has_value()) {
      j = t.value();
    } else {
      j = nullptr;
    }
  }

  template <typename T = TT>
  static std::enable_if_t<::FlexFlow::is_json_deserializable_v<T>, void>
      from_json(json const &j, std::optional<T> &t) {
    if (j == nullptr) {
      t = std::nullopt;
    } else {
      t = j.get<T>();
    }
  }
};

} // namespace nlohmann

#endif

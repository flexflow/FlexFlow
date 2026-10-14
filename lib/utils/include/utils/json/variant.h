#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_VARIANT_H

#include "utils/json/is_jsonable.h"
#include "utils/type_traits_core.h"
#include <nlohmann/json.hpp>
#include <variant>

namespace nlohmann {

template <typename... Ts>
struct adl_serializer<std::variant<Ts...>> {
  static std::enable_if_t<
      ::FlexFlow::conjunction_v<::FlexFlow::is_json_serializable<Ts>...>,
      void>
      to_json(json &j, std::variant<Ts...> const &t) {
    j["index"] = t.index();
    j["value"] = std::visit(
        [&](auto const &tt) -> nlohmann::json {
          nlohmann::json j = tt;
          return j;
        },
        t);
  }
};

} // namespace nlohmann

#endif

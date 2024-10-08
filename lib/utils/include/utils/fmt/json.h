#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_JSON_H

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace fmt {

template <typename Char>
struct formatter<::nlohmann::json, Char> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::nlohmann::json const &j, FormatContext &ctx) {
    std::ostringstream oss;
    oss << j;
    return formatter<std::string>::format(oss.str(), ctx);
  }
};

} // namespace fmt

#endif

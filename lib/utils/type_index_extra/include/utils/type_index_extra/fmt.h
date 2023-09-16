#ifndef _FLEXFLOW_LIB_UTILS_TYPE_INDEX_EXTRA_INCLUDE_UTILS_TYPE_INDEX_EXTRA_FMT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_INDEX_EXTRA_INCLUDE_UTILS_TYPE_INDEX_EXTRA_FMT_H

namespace fmt {

template <>
struct formatter<::std::type_index> : formatter<std::string> {
  template <typename FormatContext>
  auto format(std::type_index const &type_idx, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format(type_idx.name(), ctx);
  }
};

} // namespace fmt

#endif

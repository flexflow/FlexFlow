#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H

#include "utils/json.h"
#include "utils/required_core.h"
#include "utils/type_traits.h"

namespace FlexFlow {

static_assert(is_list_initializable<req<int>, int>::value, "");

}

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

namespace fmt {

template <typename T>
struct formatter<::FlexFlow::req<T>> : formatter<T> {
  template <typename FormatContext>
  auto format(::FlexFlow::req<T> const &t, FormatContext &ctx)
      -> decltype(ctx.out()) {
    return formatter<T>::format(static_cast<T>(t), ctx);
  }
};

} // namespace fmt

namespace FlexFlow {
static_assert(is_json_serializable<req<int>>::value, "");
static_assert(is_json_deserializable<req<int>>::value, "");
static_assert(is_jsonable<req<int>>::value, "");
static_assert(is_fmtable<req<int>>::value, "");
} // namespace FlexFlow

#endif

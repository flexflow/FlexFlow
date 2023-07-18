#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_SLOT_TYPE_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_SLOT_TYPE_H

#include "utils/fmt.h"

namespace FlexFlow {

enum class SlotType { TENSOR, VARIADIC };

}

namespace fmt {

template <>
struct formatter<::FlexFlow::SlotType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::SlotType d, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using ::FlexFlow::SlotType;

    string_view name = "unknown";
    switch (d) {
      case SlotType::TENSOR:
        name = "TENSOR";
        break;
      case SlotType::VARIADIC:
        name = "VARIADIC";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif

#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_CREATE_GRAD_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_CREATE_GRAD_H

#include "utils/fmt.h"

namespace FlexFlow {

enum class CreateGrad { YES, NO };

}

namespace fmt {

template <>
struct formatter<::FlexFlow::CreateGrad> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::CreateGrad ps, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (ps) {
      case CreateGrad::YES:
        name = "yes";
        break;
      case CreateGrad::NO:
        name = "no";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif

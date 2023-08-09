#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_PARAM_SYNC_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_PARAM_SYNC_H

#include "utils/fmt.h"

namespace FlexFlow {

enum class ParamSync { PS, NCCL };

}

namespace fmt {

template <>
struct formatter<::FlexFlow::ParamSync> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::ParamSync ps, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (ps) {
      case ParamSync::PS:
        name = "ParameterServer";
        break;
      case ParamSync::NCCL:
        name = "NCCL";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif

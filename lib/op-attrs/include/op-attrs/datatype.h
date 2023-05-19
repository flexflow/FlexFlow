#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H

#include "utils/fmt.h"

namespace FlexFlow {

enum class DataType {
  BOOL,
  INT32,
  INT64,
  HALF,
  FLOAT,
  DOUBLE
};

}

namespace fmt {
template <>
struct formatter<::FlexFlow::DataType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::DataType dt, FormatContext &ctx) -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (dt) {
      case DataType::BOOL: 
        name = "BOOL"; 
        break;
      case DataType::INT32: 
        name = "INT32";
        break;
      case DataType::INT64:
        name = "INT64";
        break;
      case DataType::HALF:
        name = "HALF";
        break;
      case DataType::FLOAT:
        name = "FLOAT";
        break;
      case DataType::DOUBLE:
        name = "DOUBLE";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};


}

#endif

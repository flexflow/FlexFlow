#include "pcg/file_format/v1/datatype.h"

namespace FlexFlow {

V1DataType to_v1(DataType const &dt) {
  switch (dt) {
    case DataType::BOOL:
      return V1DataType::BOOL;
    case DataType::INT32:
      return V1DataType::INT32;
    case DataType::INT64:
      return V1DataType::INT64;
    case DataType::HALF:
      return V1DataType::HALF;
    case DataType::FLOAT:
      return V1DataType::FLOAT;
    case DataType::DOUBLE:
      return V1DataType::DOUBLE;
  default:
    // Should never get here unless a new element was added to the DataType enum
    // that was not handled here.
    NOT_REACHABLE();
  }
}

V1DataTypeValue to_v1(DataTypeValue const &dv) {
  // There has to be a better way of doing this.
  if (const auto* b = get_if<bool>(&dv))
    return *b;
  else if (const auto* i32 = get_if<int32_t>(&dv))
    return *i32;
  else if (const auto* i64 = get_if<int64_t>(&dv))
    return *i64;
  else if (const auto* flt = get_if<float>(&dv))
    return *flt;
  else if (const auto* dbl = get_if<double>(&dv))
    return *dbl;
  else if (const auto* hlf = get_if<half>(&dv))
    return *hlf;
  else
    // Should never get here unless a new type was added into the DataTypeValue
    // variant which was not handled here.
    NOT_REACHABLE();
}

} // namespace FlexFlow

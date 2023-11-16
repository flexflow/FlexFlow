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
      // Should never get here unless a new element was added to the DataType
      // enum that was not handled here.
      NOT_REACHABLE();
  }
}

DataType from_v1(V1DataType const &vdt) {
  switch (vdt) {
    case V1DataType::BOOL:
      return DataType::BOOL;
    case V1DataType::INT32:
      return DataType::INT32;
    case V1DataType::INT64:
      return DataType::INT64;
    case V1DataType::HALF:
      return DataType::HALF;
    case V1DataType::FLOAT:
      return DataType::FLOAT;
    case V1DataType::DOUBLE:
      return DataType::DOUBLE;
    default:
      // Should never get here unless a new element was added to the DataType
      // enum that was not handled here.
      NOT_REACHABLE();
  }
}

V1DataTypeValue to_v1(DataTypeValue const &dv) {
  // There has to be a better way of doing this.
  if (auto const *b = get_if<bool>(&dv)) {
    return *b;
  } else if (auto const *i32 = get_if<int32_t>(&dv)) {
    return *i32;
  } else if (auto const *i64 = get_if<int64_t>(&dv)) {
    return *i64;
  } else if (auto const *flt = get_if<float>(&dv)) {
    return *flt;
  } else if (auto const *dbl = get_if<double>(&dv)) {
    return *dbl;
  } else if (auto const *hlf = get_if<half>(&dv)) {
    return *hlf;
  } else {
    // Should never get here unless a new type was added into the DataTypeValue
    // variant which was not handled here.
    NOT_REACHABLE();
  }
}

DataTypeValue from_v1(V1DataTypeValue const &vdv) {
  // There has to be a better way of doing this.
  if (auto const *b = get_if<bool>(&vdv)) {
    return *b;
  } else if (auto const *i32 = get_if<int32_t>(&vdv)) {
    return *i32;
  } else if (auto const *i64 = get_if<int64_t>(&vdv)) {
    return *i64;
  } else if (auto const *flt = get_if<float>(&vdv)) {
    return *flt;
  } else if (auto const *dbl = get_if<double>(&vdv)) {
    return *dbl;
  } else if (auto const *hlf = get_if<half>(&vdv)) {
    return *hlf;
  } else {
    // Should never get here unless a new type was added into the DataTypeValue
    // variant which was not handled here.
    NOT_REACHABLE();
  }
}

} // namespace FlexFlow

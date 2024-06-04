#include "op-attrs/datatype.h"

namespace FlexFlow {

size_t size_of_datatype(DataType data_type) {
  switch (data_type) {
    case DataType::BOOL:
      return sizeof(bool);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::HALF:
      return sizeof(float) / 2;
    case DataType::FLOAT:
      return sizeof(float);
    case DataType::DOUBLE:
      return sizeof(double);
    default:
      throw mk_runtime_error("Unknown data type {}", data_type);
  }
}

bool can_strictly_promote_datatype_from_to(DataType src, DataType dst) {
  return src < dst; 
}

} // namespace FlexFlow

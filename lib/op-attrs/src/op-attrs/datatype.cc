#include "op-attrs/datatype.h"
#include "utils/containers/contains.h"
#include "utils/exception.h"

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
      throw mk_runtime_error("Unknown DataType {}", data_type);
  }
}

bool can_strictly_promote_datatype_from_to(DataType src, DataType dst) {
  std::unordered_set<DataType> allowed;
  switch (src) {
    case DataType::BOOL:
      allowed = {
          DataType::INT32, DataType::INT64, DataType::FLOAT, DataType::DOUBLE};
      break;
    case DataType::INT32:
      allowed = {DataType::INT64};
      break;
    case DataType::INT64:
      break;
    case DataType::HALF:
      allowed = {DataType::FLOAT, DataType::DOUBLE};
      break;
    case DataType::FLOAT:
      allowed = {DataType::DOUBLE};
      break;
    case DataType::DOUBLE:
      break;
    default:
      throw mk_runtime_error(fmt::format("Unknown DataType {}", src));
  }

  return contains(allowed, dst);
}

} // namespace FlexFlow

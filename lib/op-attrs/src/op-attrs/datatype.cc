#include "op-attrs/datatype.h"
#include "utils/containers/contains.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

positive_int size_of_datatype(DataType data_type) {
  switch (data_type) {
    case DataType::BOOL:
      return positive_int{sizeof(bool)};
    case DataType::INT32:
      return positive_int{sizeof(int32_t)};
    case DataType::INT64:
      return positive_int{sizeof(int64_t)};
    case DataType::HALF:
      return positive_int{sizeof(float) / 2};
    case DataType::FLOAT:
      return positive_int{sizeof(float)};
    case DataType::DOUBLE:
      return positive_int{sizeof(double)};
    default:
      PANIC(fmt::format("Unknown DataType {}", data_type));
  }
}

bool can_strictly_promote_datatype_from_to(DataType src, DataType dst) {
  std::set<DataType> allowed;
  switch (src) {
    case DataType::BOOL:
      allowed = {DataType::INT32,
                 DataType::INT64,
                 DataType::HALF,
                 DataType::FLOAT,
                 DataType::DOUBLE};
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
      PANIC(fmt::format("Unknown DataType {}", src));
  }

  return contains(allowed, dst);
}

bool can_torch_strictly_promote_datatype_from_to(DataType src, DataType dst) {
  std::set<DataType> allowed;
  switch (src) {
    case DataType::BOOL:
      allowed = {DataType::INT32,
                 DataType::INT64,
                 DataType::HALF,
                 DataType::FLOAT,
                 DataType::DOUBLE};
      break;
    case DataType::INT32:
      allowed = {
          DataType::INT64, DataType::HALF, DataType::FLOAT, DataType::DOUBLE};
      break;
    case DataType::INT64:
      allowed = {
          DataType::INT32, DataType::HALF, DataType::FLOAT, DataType::DOUBLE};
      break;
    case DataType::HALF:
      allowed = {DataType::FLOAT, DataType::DOUBLE};
      break;
    case DataType::FLOAT:
      allowed = {DataType::HALF, DataType::DOUBLE};
      break;
    case DataType::DOUBLE:
      allowed = {DataType::HALF, DataType::FLOAT};
      break;
    default:
      PANIC(fmt::format("Unknown DataType {}", src));
  }

  return contains(allowed, dst);
}

} // namespace FlexFlow

#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H

#include "utils/fmt.h"
#include "utils/fp16.h"
#include "utils/variant.h"

namespace FlexFlow {

enum class DataType { BOOL, INT32, INT64, HALF, FLOAT, DOUBLE };

template <DataType>
struct data_type_enum_to_class;

template <>
struct data_type_enum_to_class<DataType::FLOAT> {
  using type = float;
};

template <>
struct data_type_enum_to_class<DataType::DOUBLE> {
  using type = double;
};

template <>
struct data_type_enum_to_class<DataType::INT32> {
  using type = int32_t;
};

template <>
struct data_type_enum_to_class<DataType::INT64> {
  using type = int64_t;
};

template <>
struct data_type_enum_to_class<DataType::HALF> {
  using type = half;
};

template <>
struct data_type_enum_to_class<DataType::BOOL> {
  using type = bool;
};

template <DataType DT, typename T>
typename data_type_enum_to_class<DT>::type cast_to(T t) {
  return (typename data_type_enum_to_class<DT>::type)t;
}

template <DataType DT>
using real_type = typename data_type_enum_to_class<DT>::type;

using DataTypeValue = variant<real_type<DataType::FLOAT>,
                              real_type<DataType::DOUBLE>,
                              real_type<DataType::INT32>,
                              real_type<DataType::INT64>,
                              real_type<DataType::HALF>,
                              real_type<DataType::BOOL>>;

size_t size_of(DataType);

} // namespace FlexFlow

namespace fmt {
template <>
struct formatter<::FlexFlow::DataType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::DataType dt, FormatContext &ctx)
      -> decltype(ctx.out()) {
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

} // namespace fmt

#endif

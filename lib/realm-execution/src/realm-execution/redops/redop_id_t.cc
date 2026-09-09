#include "realm-execution/redops/redop_id_t.h"
#include "utils/exception.h"

namespace FlexFlow {

redop_id_t get_sum_redop_id_for_data_type(DataType dtype) {
  switch (dtype) {
    case DataType::BOOL:
      return redop_id_t::SUM_BOOL_REDOP_ID;
    case DataType::INT32:
      return redop_id_t::SUM_INT32_REDOP_ID;
    case DataType::INT64:
      return redop_id_t::SUM_INT64_REDOP_ID;
    case DataType::FLOAT:
      return redop_id_t::SUM_FLOAT_REDOP_ID;
    case DataType::DOUBLE:
      return redop_id_t::SUM_DOUBLE_REDOP_ID;
    default:
      PANIC("No known sum reduction for data type {}", dtype);
  }
}

Realm::ReductionOpID
    get_realm_reduction_op_id_for_redop_id(redop_id_t redop_id) {
  return static_cast<Realm::ReductionOpID>(redop_id) + 1;
}

} // namespace FlexFlow

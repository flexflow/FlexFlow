#include "op-attrs/ops/cast.h"

namespace FlexFlow {

CastAttrs::CastAttrs(DataType _data_type) : dtype(_data_type) {}

/* bool CastAttrs::is_valid(ParallelTensorShape const &input) const { */
/*   bool valid = input.is_valid(); */
/*   valid &= (input.at(input.num_dims() - 1).degree == 1); */
/*   return valid; */
/* } */

} // namespace FlexFlow

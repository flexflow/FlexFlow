#include "op-attrs/ops/cast.h"
#include "op-attrs/datatype.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_output_shape(CastAttrs const &attrs, TensorShape const &input) {

  if (!can_strictly_promote_datatype_from_to(input.data_type, attrs.dtype)) {
    return tl::unexpected(fmt::format(
        "Cast cannot strictly promote input datatype {} to output datatype {}",
        input.data_type,
        attrs.dtype));
  }

  TensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(CastAttrs const &attrs, ParallelTensorShape const &input) {

  if (!can_strictly_promote_datatype_from_to(input.data_type, attrs.dtype)) {
    return tl::unexpected(fmt::format(
        "Cast cannot strictly promote input datatype {} to output datatype {}",
        input.data_type,
        attrs.dtype));
  }

  ParallelTensorShape output = input;
  output.data_type = attrs.dtype;

  return output;
}

/* bool CastAttrs::is_valid(ParallelTensorShape const &input) const { */
/*   bool valid = input.is_valid(); */
/*   valid &= (input.at(input.num_dims() - 1).degree == 1); */
/*   return valid; */
/* } */

} // namespace FlexFlow

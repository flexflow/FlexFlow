#include "op-attrs/ops/attention.h"

namespace FlexFlow {

/* bool MultiHeadAttentionAttrs::is_valid(std::vector<ParallelTensorShape> const
 * &inputs) const { */
/*   return (inputs.size() == 3 && std::all_of(inputs.begin(), inputs.end(),
 * [](ParallelTensorShape const &s) { return s.is_valid(); })); */
/*   bool is_valid = true; */
/*   return is_valid; */
/* } */

int get_qProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

int get_vProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.vdim;
}

int get_kProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

int get_oProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.embed_dim;
}

int get_qSize(TensorShape const &query_shape) {
  return query_shape.at(ff_dim_t(0));
}

int get_kSize(TensorShape const &key_shape) {
  return key_shape.at(ff_dim_t(0));
}

int get_vSize(TensorShape const &value_shape) {
  return value_shape.at(ff_dim_t(0));
}

TensorShape
    get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      MultiHeadAttentionInputs<TensorShape> const &inputs) {
  size_t qParas = get_qProjSize(attrs) * get_qSize(inputs);
  size_t kParas = get_kProjSize(attrs) * get_kSize(inputs);
  size_t vParas = get_vProjSize(attrs) * get_vSize(inputs);
  TensorShape output_shape = get_output_shape(attrs, inputs);
  size_t oParas = get_oProjSize(attrs) * get_oSize(output_shape);

  TensorDims dims = {qParas + kParas + vParas + oParas,
                     static_cast<size_t>(attrs.embed_dim)};

  return {dims, DataType::FLOAT};
}

ParallelTensorShape get_output_shape(MultiHeadAttentionAttrs const &attrs,
                                     ParallelTensorShape const &query_shape,
                                     ParallelTensorShape const &key_shape,
                                     ParallelTensorShape const &value_shape) {
  /* ParallelDim replica_dim = query_shape.at(ff_dim_t(query_shape.num_dims() -
   * 2)); */
  /* replica_dim.size = replica_dim.degree; */

  /* ParallelDim */

  ParallelTensorShape output_shape = query_shape;
  output_shape.at(ff_dim_t(output_shape.num_dims() - 1)).size = attrs.embed_dim;
  return output_shape;
}

TensorShape get_output_shape(MultiHeadAttentionAttrs const &attrs,
                             TensorShape const &query_shape,
                             TensorShape const &key_shape,
                             TensorShape const &value_shape) {
  ParallelTensorShape parallel_shape =
      get_output_shape(attrs,
                       static_cast<ParallelTensorShape>(query_shape),
                       static_cast<ParallelTensorShape>(key_shape),
                       static_cast<ParallelTensorShape>(value_shape));
  return get_tensor_shape_unsafe(parallel_shape);
}

} // namespace FlexFlow

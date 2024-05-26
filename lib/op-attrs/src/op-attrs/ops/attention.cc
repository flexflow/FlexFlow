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
  return dim_at_idx(query_shape, ff_dim_t(0));
}

int get_kSize(TensorShape const &key_shape) {
  return dim_at_idx(key_shape, ff_dim_t(0));
}

int get_vSize(TensorShape const &value_shape) {
  return dim_at_idx(value_shape, ff_dim_t(0));
}

int get_qSize(ParallelMultiHeadAttentionInputs const &) {
  NOT_IMPLEMENTED();
}

int get_qSize(MultiHeadAttentionInputs const &) {
  NOT_IMPLEMENTED();
}

int get_kSize(ParallelMultiHeadAttentionInputs const &) {
  NOT_IMPLEMENTED();
}

int get_kSize(MultiHeadAttentionInputs const &) {
  NOT_IMPLEMENTED();
}

int get_vSize(ParallelMultiHeadAttentionInputs const &) {
  NOT_IMPLEMENTED();
}

int get_vSize(MultiHeadAttentionInputs const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                              MultiHeadAttentionInputs const &inputs) {
  size_t qParas = get_qProjSize(attrs) * get_qSize(inputs);
  size_t kParas = get_kProjSize(attrs) * get_kSize(inputs);
  size_t vParas = get_vProjSize(attrs) * get_vSize(inputs);
  TensorShape output_shape = get_output_shape(attrs, inputs);
  size_t oParas = get_oProjSize(attrs) * get_oSize(output_shape);

  TensorDims dims = {{qParas + kParas + vParas + oParas,
                      static_cast<size_t>(attrs.embed_dim)}};

  return {dims, DataType::FLOAT};
}

ParallelTensorShape get_output_shape(MultiHeadAttentionAttrs const &attrs,
                                     ParallelTensorShape const &query_shape,
                                     ParallelTensorShape const &key_shape,
                                     ParallelTensorShape const &value_shape) {
  NOT_IMPLEMENTED();
  /* ParallelTensorShape output_shape = query_shape; */
  /* dim_at_idx(output_shape, ff_dim_t(num_dims(output_shape) - 1)).size = attrs.embed_dim; */
  /* return output_shape; */
}

TensorShape get_output_shape(MultiHeadAttentionAttrs const &attrs,
                             TensorShape const &query_shape,
                             TensorShape const &key_shape,
                             TensorShape const &value_shape) {
  ParallelTensorShape parallel_shape =
      get_output_shape(attrs,
                       lift_to_parallel(query_shape),
                       lift_to_parallel(key_shape),
                       lift_to_parallel(value_shape));
  return get_tensor_shape_unsafe(parallel_shape);
}
TensorShape get_output_shape(MultiHeadAttentionAttrs const &,
                             MultiHeadAttentionInputs const &) {
  NOT_IMPLEMENTED();
}

int get_oSize(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

int get_oSize(TensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

// Tensor FFModel::multihead_attention(const Tensor query,
//                                     const Tensor key,
//                                     const Tensor value,
//                                     int embed_dim,
//                                     int num_heads,
//                                     int kdim,
//                                     int vdim,
//                                     float dropout,
//                                     bool bias,
//                                     bool add_bias_kv,
//                                     bool add_zero_attn,
//                                     Initializer *kernel_initializer,
//                                     char const *name) {
//   Layer *li = new Layer(this,
//                         OP_MULTIHEAD_ATTENTION,
//                         DT_FLOAT,
//                         name,
//                         3 /*inputs*/,
//                         1 /*weights*/,
//                         1 /*outputs*/,
//                         query,
//                         key,
//                         value);
//   {
//     int numdims = query->num_dims;
//     int dims[MAX_TENSOR_DIM];
//     for (int i = 0; i < numdims; i++) {
//       dims[i] = query->dims[i];
//     }
//     dims[0] = embed_dim;
//     li->outputs[0] = create_tensor_legion_ordering(
//         numdims, dims, DT_FLOAT, li, 0, true /*create_grad*/);
//   }
//   {
//     // Compute weight size
//     int qProjSize = kdim, kProjSize = kdim, vProjSize = kdim,
//         oProjSize = embed_dim;
//     int qSize = query->dims[0], kSize = key->dims[0], vSize = value->dims[0];
//     int qParas = qProjSize * qSize;
//     int kParas = kProjSize * kSize;
//     int vParas = vProjSize * vSize;
//     int oParas = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
//     int dims[2] = {qParas + kParas + vParas + oParas, num_heads};
//     li->weights[0] = create_weight_legion_ordering(2,
//                                                    dims,
//                                                    DT_FLOAT,
//                                                    li,
//                                                    true /*create_grad*/,
//                                                    kernel_initializer,
//                                                    CHOSEN_SYNC_TYPE);
//   }
//   li->data_type = DT_FLOAT;
//   li->add_int_property("embed_dim", embed_dim);
//   li->add_int_property("num_heads", num_heads);
//   li->add_int_property("kdim", kdim);
//   li->add_int_property("vdim", vdim);
//   li->add_int_property("bias", bias);
//   li->add_int_property("add_bias_kv", add_bias_kv);
//   li->add_int_property("add_zero_attn", add_zero_attn);
//   li->add_float_property("dropout", dropout);
//   layers.push_back(li);
//   return li->outputs[0];
// }

// MultiHeadAttention::MultiHeadAttention(FFModel &model,
//                                        LayerID const &_layer_guid,
//                                        const ParallelTensor _query,
//                                        const ParallelTensor _key,
//                                        const ParallelTensor _value,
//                                        int _embed_dim,
//                                        int _num_heads,
//                                        int _kdim,
//                                        int _vdim,
//                                        float _dropout,
//                                        bool _bias,
//                                        bool _add_bias_kv,
//                                        bool _add_zero_attn,
//                                        bool allocate_weights,
//                                        char const *name)
//     // Initializer* _bias_initializer)
//     : Op(model,
//          OP_MULTIHEAD_ATTENTION,
//          DT_FLOAT,
//          name,
//          3 /*inputs*/,
//          1 /*weights*/,
//          1 /*outputs*/,
//          _query,
//          _key,
//          _value),
//       attrs(_embed_dim,
//             _num_heads,
//             _kdim,
//             _vdim,
//             _dropout,
//             _bias,
//             _add_bias_kv,
//             _add_zero_attn),
//       qSize(_query->dims[0].size), kSize(_key->dims[0].size),
//       vSize(_value->dims[0].size), qProjSize(_kdim),
//       qoSeqLength(_query->dims[1].size), kvSeqLength(_key->dims[1].size) {
//   // overwrite layer_guid
//   layer_guid = _layer_guid;

//   // assert key and value have the same sequence length
//   assert(_key->dims[1] == _value->dims[1]);
//   numOutputs = 1;
//   int numdim = _query->num_dims;
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < numdim; i++) {
//     dims[i] = _query->dims[i];
//   }
//   dims[0].size = _embed_dim;
//   // Currently require no parallelism along this dim
//   assert(dims[0].degree == 1);
//   if (allocate_weights) {
//     // Create weight tensor
//     int num_dims = inputs[0]->num_dims;
//     // Compute weight size
//     int qParas = this->qProjSize * this->qSize;
//     int kParas = kProjSize(attrs) * this->kSize;
//     int vParas = vProjSize(attrs) * this->vSize;
//     int oParas = oProjSize(attrs) *
//                  (vProjSize(attrs) > 0 ? vProjSize(attrs) : this->vSize);
//     ParallelDim dims[3];
//     dims[0] = inputs[0]->dims[num_dims - 2];
//     dims[0].size = dims[0].degree;
//     dims[1] = inputs[0]->dims[num_dims - 1];
//     dims[1].size = this->attrs.num_heads;
//     dims[2].size = qParas + kParas + vParas + oParas;
//     dims[2].degree = 1;
//     dims[2].parallel_idx = -1;
//     int seed = std::rand();
//     Initializer *initializer = new GlorotUniform(seed);
// #ifdef USE_NCCL
//     ParameterSyncType comm_type = ParameterSyncType::NCCL;
// #else
//     ParameterSyncType comm_type = ParameterSyncType::PS;
// #endif
//     weights[0] = model.create_parallel_weight<3>(dims,
//                                                  DT_FLOAT,
//                                                  NULL /*owner_op*/,
//                                                  true /*create_grad*/,
//                                                  initializer,
//                                                  comm_type);
//   }

//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       _query->num_dims, dims, DT_FLOAT, this);
//   /* for (int i = 0; i < numdim; i++) { */
//   /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
//   /* } */
//   /* // Check correctness */
//   /* assert(check_output_input_weight_parallel_dims()); */
// }

// MultiHeadAttention::MultiHeadAttention(FFModel &model,
//                                        const ParallelTensor _query,
//                                        const ParallelTensor _key,
//                                        const ParallelTensor _value,
//                                        const ParallelTensor _weight,
//                                        int _embed_dim,
//                                        int _num_heads,
//                                        int _kdim,
//                                        int _vdim,
//                                        float _dropout,
//                                        bool _bias,
//                                        bool _add_bias_kv,
//                                        bool _add_zero_attn,
//                                        bool allocate_weights,
//                                        char const *name)
//     // Initializer* _bias_initializer)
//     : Op(model,
//          OP_MULTIHEAD_ATTENTION,
//          DT_FLOAT,
//          name,
//          3 /*inputs*/,
//          1 /*weights*/,
//          1 /*outputs*/,
//          _query,
//          _key,
//          _value,
//          _weight),
//       attrs(_embed_dim,
//             _num_heads,
//             _kdim,
//             _vdim,
//             _dropout,
//             _bias,
//             _add_bias_kv,
//             _add_zero_attn),
//       qSize(_query->dims[0].size), kSize(_key->dims[0].size),
//       vSize(_value->dims[0].size), qProjSize(_kdim),
//       qoSeqLength(_query->dims[1].size), kvSeqLength(_key->dims[1].size)
// // bias_initializer(_bias_initializer)
// {
//   // assert key and value have the same sequence length
//   assert(_key->dims[1] == _value->dims[1]);
//   numOutputs = 1;
//   int numdim = _query->num_dims;
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < numdim; i++) {
//     dims[i] = _query->dims[i];
//   }
//   // assert key and value have the same sequence length
//   assert(_key->dims[1] == _value->dims[1]);
//   dims[0].size = _embed_dim;
//   // Currently require no parallelism along this dim
//   assert(dims[0].degree == 1);
//   if (allocate_weights) {
//     // Create weight tensor
//     int num_dims = inputs[0]->num_dims;
//     // Compute weight size
//     int qParas = this->qProjSize * this->qSize;
//     int kParas = kProjSize(attrs) * this->kSize;
//     int vParas = vProjSize(attrs) * this->vSize;
//     int oParas = oProjSize(attrs) *
//                  (vProjSize(attrs) > 0 ? vProjSize(attrs) : this->vSize);
//     ParallelDim dims[3];
//     dims[0] = inputs[0]->dims[num_dims - 2];
//     dims[0].size = dims[0].degree;
//     dims[1] = inputs[0]->dims[num_dims - 1];
//     dims[1].size = this->attrs.num_heads;
//     dims[2].size = qParas + kParas + vParas + oParas;
//     int seed = std::rand();
//     Initializer *initializer = new GlorotUniform(seed);
// #ifdef USE_NCCL
//     ParameterSyncType comm_type = ParameterSyncType::NCCL;
// #else
//     ParameterSyncType comm_type = ParameterSyncType::PS;
// #endif
//     weights[0] = model.create_parallel_weight<3>(dims,
//                                                  DT_FLOAT,
//                                                  NULL /*owner_op*/,
//                                                  true /*create_grad*/,
//                                                  initializer,
//                                                  comm_type);
//   }
//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       _query->num_dims, dims, DT_FLOAT, this);

//   /* for (int i = 0; i < numdim; i++) { */
//   /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
//   /* } */
//   /* register_output_weight_parallel_dims(outputs[0], numdim-1, _weight, 1);
//   */
//   /* register_output_weight_parallel_dims(outputs[0], numdim-2, _weight, 2);
//   */
//   // Check correctness
//   /* assert(check_output_input_weight_parallel_dims()); */
// }

// void MultiHeadAttention::forward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_forward(ff, argmap);
//   int idx = 0;
//   IndexLauncher launcher(ATTENTION_FWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(NULL, 0),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(idx++, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[1]->region));
//   launcher.add_field(idx++, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[2]->region));
//   launcher.add_field(idx++, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(weights[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     weights[0]->region));
//   launcher.add_field(idx++, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(4, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

// void MultiHeadAttention::backward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_backward(ff, argmap);
//   IndexLauncher launcher(ATTENTION_BWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(NULL, 0),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[1]->region));
//   launcher.add_field(1, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[2]->region));
//   launcher.add_field(2, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(weights[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     weights[0]->region));
//   launcher.add_field(3, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region_grad));
//   launcher.add_field(4, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     weights[0]->region_grad));
//   launcher.add_field(5, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region_grad));
//   launcher.add_field(6, FID_DATA);
//   int num_regions = 7;
//   if (inputs[1]->region != inputs[0]->region) {
//     // when key != query
//     launcher.add_region_requirement(RegionRequirement(inputs[1]->part_grad,
//                                                       0 /*projection id*/,
//                                                       READ_WRITE,
//                                                       EXCLUSIVE,
//                                                       inputs[1]->region_grad));
//     launcher.add_field(num_regions++, FID_DATA);
//   }
//   if ((inputs[2]->region != inputs[0]->region) &&
//       (inputs[2]->region != inputs[1]->region)) {
//     // when value != key and value != query
//     launcher.add_region_requirement(RegionRequirement(inputs[2]->part_grad,
//                                                       0 /*projection id*/,
//                                                       READ_WRITE,
//                                                       EXCLUSIVE,
//                                                       inputs[2]->region_grad));
//     launcher.add_field(num_regions++, FID_DATA);
//   }
//   runtime->execute_index_space(ctx, launcher);
// }

#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/extend.h"
#include "utils/integer_conversions.h"

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

int get_qSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.query_dim.size;
}

int get_qSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.query_size;
}

int get_kSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.key_dim.size;
}

int get_kSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.key_size;
}

int get_vSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.value_dim.size;
}

int get_vSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.value_size;
}

int get_kvSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size;
}

int get_kvSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length;
}

int get_qoSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size; // FIXME -- assumes only prefill
}

int get_qoSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length; // FIXME -- assumes only prefil
}

int get_num_samples(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.batch_dim.size;
}

int get_num_samples(MultiHeadAttentionInputs const &inputs) {
  return inputs.batch_size;
}

std::vector<IncomingTensorRole>
    get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs const &attrs) {

  std::vector<IncomingTensorRole> roles = std::vector{
      IncomingTensorRole::INPUT,
      IncomingTensorRole::INPUT,
      IncomingTensorRole::INPUT,
      IncomingTensorRole::WEIGHT,
  };

  if (attrs.bias) {
    extend(roles,
           std::vector{IncomingTensorRole::WEIGHT, IncomingTensorRole::WEIGHT});
  }

  return roles;
}

tl::expected<TensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &attrs,
                     TensorShape const &input_q,
                     TensorShape const &input_k,
                     TensorShape const &input_v) {
  tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
      parse_attention_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }

  MultiHeadAttentionInputs parsed = parse_result.value();

  return TensorShape{
      TensorDims{FFOrdered<size_t>{
          parsed.batch_size,
          parsed.sequence_length,
          size_t_from_int(attrs.embed_dim),
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v) {
  tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
      parse_attention_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }

  MultiHeadAttentionInputs parsed = parse_result.value();

  // W^Q_i in "Attention Is All You Need" top of page 5
  size_t qProjectWeightSize = parsed.query_size * attrs.kdim;

  // W^K_i in "Attention Is All You Need" top of page 5 (all i's put together)
  size_t kProjectWeightSize = parsed.key_size * attrs.kdim;

  // W^V_i in "Attention Is All You Need" top of page 5 (all i's put together)
  size_t vProjectWeightSize = parsed.value_size * attrs.vdim;

  // W^O in "Attention Is All You Need" top of page 5, with num_heads factored
  // out
  size_t outWeightSize = attrs.vdim * attrs.embed_dim;

  return TensorShape{
      TensorDims{FFOrdered<size_t>{
          (qProjectWeightSize + kProjectWeightSize + vProjectWeightSize +
           outWeightSize),
          size_t_from_int(attrs.num_heads),
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &attrs,
                         TensorShape const &input_q,
                         TensorShape const &input_k,
                         TensorShape const &input_v) {
  MultiHeadAttentionInputs parsed = ({
    tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
        parse_attention_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }
    parse_result.value();
  });

  return TensorShape{
      TensorDims{FFOrdered<size_t>{
          size_t_from_int(attrs.kdim + attrs.kdim + attrs.vdim),
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &attrs,
                          TensorShape const &input_q,
                          TensorShape const &input_k,
                          TensorShape const &input_v) {
  MultiHeadAttentionInputs parsed = ({
    tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
        parse_attention_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }
    parse_result.value();
  });

  return TensorShape{
      TensorDims{FFOrdered<size_t>{
          size_t_from_int(attrs.embed_dim),
      }},
      parsed.datatype,
  };
}

tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v) {
  tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }
  MultiHeadAttentionParallelInputs parsed = parse_result.value();

  tl::expected<TensorShape, std::string> result_unpar_get_shape =
      get_weights_shape(attrs,
                        get_reduced_shape(input_q),
                        get_reduced_shape(input_k),
                        get_reduced_shape(input_v));
  if (!result_unpar_get_shape.has_value()) {
    return tl::unexpected(result_unpar_get_shape.error());
  }
  TensorShape unpar_shape = result_unpar_get_shape.value();

  int joined_dim_degree = 1;
  int head_dim_degree = parsed.discard_copy_degree.value;

  return lift_to_parallel_with_degrees(
      unpar_shape,
      SumDegree{1},
      DiscardCopyDegree{parsed.batch_dim.degree},
      FFOrdered<int>{joined_dim_degree, head_dim_degree});
}

tl::expected<ParallelTensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &attrs,
                         ParallelTensorShape const &input_q,
                         ParallelTensorShape const &input_k,
                         ParallelTensorShape const &input_v) {
  MultiHeadAttentionParallelInputs parsed = ({
    tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }

    parse_result.value();
  });

  TensorShape unpar_shape = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_input_bias_shape(attrs,
                             get_reduced_shape(input_q),
                             get_reduced_shape(input_k),
                             get_reduced_shape(input_v));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }

    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<int> shard_degrees = FFOrdered<int>{1};
  return lift_to_parallel_with_degrees(
      unpar_shape, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &attrs,
                          ParallelTensorShape const &input_q,
                          ParallelTensorShape const &input_k,
                          ParallelTensorShape const &input_v) {
  MultiHeadAttentionParallelInputs parsed = ({
    tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }

    parse_result.value();
  });

  TensorShape unpar_shape = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_bias_shape(attrs,
                              get_reduced_shape(input_q),
                              get_reduced_shape(input_k),
                              get_reduced_shape(input_v));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }

    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<int> shard_degrees = FFOrdered<int>{1};
  return lift_to_parallel_with_degrees(
      unpar_shape, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &attrs,
                     ParallelTensorShape const &input_q,
                     ParallelTensorShape const &input_k,
                     ParallelTensorShape const &input_v) {
  tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }
  MultiHeadAttentionParallelInputs parsed = parse_result.value();

  tl::expected<TensorShape, std::string> result_unpar_get_shape =
      get_output_shape(attrs,
                       get_reduced_shape(input_q),
                       get_reduced_shape(input_k),
                       get_reduced_shape(input_v));
  if (!result_unpar_get_shape.has_value()) {
    return tl::unexpected(result_unpar_get_shape.error());
  }
  TensorShape unpar_shape = result_unpar_get_shape.value();

  int sum_degree = parsed.discard_copy_degree.value;
  int discard_copy_degree = 1;
  int batch_degree = parsed.batch_dim.degree;
  int seq_len_degree = 1;
  int out_dim_degree = 1;

  return lift_to_parallel_with_degrees(
      unpar_shape,
      SumDegree{sum_degree},
      DiscardCopyDegree{discard_copy_degree},
      FFOrdered<int>{batch_degree, seq_len_degree, out_dim_degree});
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

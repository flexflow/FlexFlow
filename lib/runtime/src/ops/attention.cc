/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "attention.h"
#include "kernels/attention_kernels.h"
#include "kernels/profiling.h"
#include "realm_allocator.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::MultiHeadAttention;

enum Slots {
  ATTRS,
  PROFILING,
  QUERY,
  KEY,
  VALUE,
  WEIGHTS,
  OUTPUT,
  PER_DEVICE_STATE
};

OpTaskInvocation init(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0));
  b.bind(KEY, input_tensor(1));
  b.bind(VALUE, input_tensor(2));
  b.bind(WEIGHTS, param_tensor(0));
  b.bind(OUTPUT, output_tensor(0));

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, enable_profiling());

  return {ATTENTION_INIT_TASK_ID, b};
}

OpTaskInvocation forward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0));
  b.bind(KEY, input_tensor(1));
  b.bind(VALUE, input_tensor(2));
  b.bind(WEIGHTS, param_tensor(0));
  b.bind(OUTPUT, param_tensor(0));
  b.bind_arg(PER_DEVICE_STATE, per_device_op_state<MHAPerDeviceState>());

  return {ATTENTION_FWD_TASK_ID, b};
}

OpTaskInvocation backward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0));
  b.bind(KEY, input_tensor(1));
  b.bind(VALUE, input_tensor(2));
  b.bind_grad(QUERY, input_tensor(0));
  b.bind_grad(KEY, input_tensor(1));
  b.bind_grad(VALUE, input_tensor(2));
  b.bind(WEIGHTS, param_tensor(0));
  b.bind_grad(WEIGHTS, param_tensor(0));
  b.bind_grad(OUTPUT, output_tensor(0));

  return {ATTENTION_BWD_TASK_ID, b};
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  MultiHeadAttentionAttrs const &attrs,
                                  ParallelTensorShape const &query_shape,
                                  ParallelTensorShape const &key_shape,
                                  ParallelTensorShape const &value_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  auto query = allocate_input(env, query_shape);
  auto key = allocate_input(env, key_shape);
  auto value = allocate_input(env, value_shape);

  auto query_grad = allocate_input_grad(env, query_shape);
  auto key_grad = allocate_input_grad(env, key_shape);
  auto value_grad = allocate_input_grad(env, value_shape);

  MultiHeadAttentionInputs<ParallelTensorShape> input_shapes = {
      query_shape, key_shape, value_shape};

  ParallelTensorShape weight_shape = get_weights_shape(attrs, input_shapes);
  auto weights = allocate_weight(env, weight_shape);
  auto weights_grad = allocate_weight_grad(env, weight_shape);
  ParallelTensorShape output_shape = get_output_shape(attrs, input_shapes);
  auto output = allocate_output(env, output_shape);
  auto output_grad = allocate_output_grad(env, output_shape);

  MHAPerDeviceState per_device_state = init_kernel(
      get_ff_handle(env), create_allocator(env), get_num_samples(input_shapes),
      attrs.num_heads, get_qSize(input_shapes), get_kSize(input_shapes),
      get_vSize(input_shapes), get_qProjSize(attrs), get_kProjSize(attrs),
      get_vProjSize(attrs), get_oProjSize(attrs), get_qoSeqLength(input_shapes),
      get_kvSeqLength(input_shapes), attrs.add_bias_kv);

  float forward_time =
      profiling_wrapper(forward_kernel, settings, get_float_ptr(query),
                        get_float_ptr(key), get_float_ptr(value),
                        get_float_ptr(weights), get_float_ptr(output))
          .value();

  float backward_time =
      profiling_wrapper(backward_kernel, settings, get_float_ptr(query),
                        get_float_ptr(query_grad), get_float_ptr(key),
                        get_float_ptr(key_grad), get_float_ptr(value),
                        get_float_ptr(value_grad), get_float_ptr(weights),
                        get_float_ptr(weights_grad), get_float_ptr(output_grad))
          .value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}

Tensor FFModel::multihead_attention(const Tensor query, const Tensor key,
                                    const Tensor value, int embed_dim,
                                    int num_heads, int kdim, int vdim,
                                    float dropout, bool bias, bool add_bias_kv,
                                    bool add_zero_attn,
                                    Initializer *kernel_initializer,
                                    char const *name) {
  Layer *li =
      new Layer(this, OP_MULTIHEAD_ATTENTION, DT_FLOAT, name, 3 /*inputs*/,
                1 /*weights*/, 1 /*outputs*/, query, key, value);
  {
    int numdims = query->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = query->dims[i];
    }
    dims[0] = embed_dim;
    li->outputs[0] = create_tensor_legion_ordering(numdims, dims, DT_FLOAT, li,
                                                   0, true /*create_grad*/);
  }
  {
    // Compute weight size
    int qProjSize = kdim, kProjSize = kdim, vProjSize = kdim,
        oProjSize = embed_dim;
    int qSize = query->dims[0], kSize = key->dims[0], vSize = value->dims[0];
    int qParas = qProjSize * qSize;
    int kParas = kProjSize * kSize;
    int vParas = vProjSize * vSize;
    int oParas = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
    int dims[2] = {qParas + kParas + vParas + oParas, num_heads};
    li->weights[0] = create_weight_legion_ordering(
        2, dims, DT_FLOAT, li, true /*create_grad*/, kernel_initializer,
        CHOSEN_SYNC_TYPE);
  }
  li->data_type = DT_FLOAT;
  li->add_int_property("embed_dim", embed_dim);
  li->add_int_property("num_heads", num_heads);
  li->add_int_property("kdim", kdim);
  li->add_int_property("vdim", vdim);
  li->add_int_property("bias", bias);
  li->add_int_property("add_bias_kv", add_bias_kv);
  li->add_int_property("add_zero_attn", add_zero_attn);
  li->add_float_property("dropout", dropout);
  layers.push_back(li);
  return li->outputs[0];
}

MultiHeadAttention::MultiHeadAttention(
    FFModel &model, LayerID const &_layer_guid, const ParallelTensor _query,
    const ParallelTensor _key, const ParallelTensor _value, int _embed_dim,
    int _num_heads, int _kdim, int _vdim, float _dropout, bool _bias,
    bool _add_bias_kv, bool _add_zero_attn, bool allocate_weights,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model, OP_MULTIHEAD_ATTENTION, DT_FLOAT, name, 3 /*inputs*/,
         1 /*weights*/, 1 /*outputs*/, _query, _key, _value),
      attrs(_embed_dim, _num_heads, _kdim, _vdim, _dropout, _bias, _add_bias_kv,
            _add_zero_attn),
      qSize(_query->dims[0].size), kSize(_key->dims[0].size),
      vSize(_value->dims[0].size), qProjSize(_kdim),
      qoSeqLength(_query->dims[1].size), kvSeqLength(_key->dims[1].size) {
  // overwrite layer_guid
  layer_guid = _layer_guid;

  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  numOutputs = 1;
  int numdim = _query->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _query->dims[i];
  }
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);
  if (allocate_weights) {
    // Create weight tensor
    int num_dims = inputs[0]->num_dims;
    // Compute weight size
    int qParas = this->qProjSize * this->qSize;
    int kParas = kProjSize(attrs) * this->kSize;
    int vParas = vProjSize(attrs) * this->vSize;
    int oParas = oProjSize(attrs) *
                 (vProjSize(attrs) > 0 ? vProjSize(attrs) : this->vSize);
    ParallelDim dims[3];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->attrs.num_heads;
    dims[2].size = qParas + kParas + vParas + oParas;
    dims[2].degree = 1;
    dims[2].parallel_idx = -1;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
#ifdef USE_NCCL
    ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
    ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
    weights[0] = model.create_parallel_weight<3>(
        dims, DT_FLOAT, NULL /*owner_op*/, true /*create_grad*/, initializer,
        comm_type);
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _query->num_dims, dims, DT_FLOAT, this);
  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* // Check correctness */
  /* assert(check_output_input_weight_parallel_dims()); */
}

MultiHeadAttention::MultiHeadAttention(
    FFModel &model, const ParallelTensor _query, const ParallelTensor _key,
    const ParallelTensor _value, const ParallelTensor _weight, int _embed_dim,
    int _num_heads, int _kdim, int _vdim, float _dropout, bool _bias,
    bool _add_bias_kv, bool _add_zero_attn, bool allocate_weights,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model, OP_MULTIHEAD_ATTENTION, DT_FLOAT, name, 3 /*inputs*/,
         1 /*weights*/, 1 /*outputs*/, _query, _key, _value, _weight),
      attrs(_embed_dim, _num_heads, _kdim, _vdim, _dropout, _bias, _add_bias_kv,
            _add_zero_attn),
      qSize(_query->dims[0].size), kSize(_key->dims[0].size),
      vSize(_value->dims[0].size), qProjSize(_kdim),
      qoSeqLength(_query->dims[1].size), kvSeqLength(_key->dims[1].size)
// bias_initializer(_bias_initializer)
{
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  numOutputs = 1;
  int numdim = _query->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _query->dims[i];
  }
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);
  if (allocate_weights) {
    // Create weight tensor
    int num_dims = inputs[0]->num_dims;
    // Compute weight size
    int qParas = this->qProjSize * this->qSize;
    int kParas = kProjSize(attrs) * this->kSize;
    int vParas = vProjSize(attrs) * this->vSize;
    int oParas = oProjSize(attrs) *
                 (vProjSize(attrs) > 0 ? vProjSize(attrs) : this->vSize);
    ParallelDim dims[3];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->attrs.num_heads;
    dims[2].size = qParas + kParas + vParas + oParas;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
#ifdef USE_NCCL
    ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
    ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
    weights[0] = model.create_parallel_weight<3>(
        dims, DT_FLOAT, NULL /*owner_op*/, true /*create_grad*/, initializer,
        comm_type);
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _query->num_dims, dims, DT_FLOAT, this);

  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* register_output_weight_parallel_dims(outputs[0], numdim-1, _weight, 1); */
  /* register_output_weight_parallel_dims(outputs[0], numdim-2, _weight, 2); */
  // Check correctness
  /* assert(check_output_input_weight_parallel_dims()); */
}

static PerDeviceOpState *init_task(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx, Runtime *runtime) {
  OpTaskArgumentAccessor acc(task, regions, ctx, runtime);

  auto const &attrs = acc.get_argument<MultiHeadAttentionAttrs>(ATTRS);
  bool profiling = acc.get_argument<bool>(PROFILING);
  int kvSeqLength = acc.get_argument<int>(KVSEQLENGTH);
  int kSize = acc.get_argument<int>(KSIZE);
  int vSize = acc.get_argument<int>(VSIZE);
  int qProjSize = acc.get_argument<int>(QPROJSIZE);

  FFHandler handle = *((FFHandler const *)task->local_args);

  auto query = acc.get_tensor<READ_ONLY>(QUERY);
  auto key = acc.get_tensor<READ_ONLY>(KEY);
  auto value = acc.get_tensor<READ_ONLY>(VALUE);
  auto weight = acc.get_tensor<READ_ONLY>(WEIGHTS);
  auto output = acc.get_tensor<WRITE_ONLY>(OUTPUT);

  int qoSeqLength = query.dims[legion_dim_t(1)];

  int num_samples = query.shape[2];
  assert(qoSeqLength == query.shape[1]);
  assert(qSize == query.shape[0]);
  assert(num_samples == key.shape[2]);
  assert(kvSeqLength == key.shape[1]);
  assert(kSize == key.shape[0]);
  assert(num_samples == value.shape[2]);
  assert(kvSeqLength == value.shape[1]);
  assert(vSize == value.shape[0]);
  int num_heads = weight.shape[1];
  assert(num_samples == output.shape[2]);
  assert(qoSeqLength == output.shape[1]);
  assert(oProjSize(attrs) == output.shape[0]);

  MHAPerDeviceState *m = new MHAPerDeviceState(
      handle, get_gpu_memory_allocator(task), num_samples, num_heads, qSize,
      kSize, vSize, qProjSize, kProjSize(attrs), vProjSize(attrs),
      oProjSize(attrs), qoSeqLength, kvSeqLength, attrs.add_bias_kv);

  m->profiling = profiling;
  assert(weight.shape.get_volume() * sizeof(float) == m->weightSize);
  return m;
}

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

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx, Runtime *runtime) {
  OpTaskArgumentAccessor acc(task, regions, ctx, runtime);

  auto query = acc.get_tensor<READ_ONLY>(QUERY);
  auto key = acc.get_tensor<READ_ONLY>(KEY);
  auto value = acc.get_tensor<READ_ONLY>(VALUE);
  auto weight = acc.get_tensor<READ_ONLY>(WEIGHTS);
  auto output = acc.get_tensor<WRITE_ONLY>(OUTPUT);
  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);

  profile(forward_kernel, m->profiling,
          "[MultiHeadAttention] forward_time = %.2lfms\n", m,
          query.get_float_ptr(), key.get_float_ptr(), value.get_float_ptr(),
          weight.get_float_ptr(), output.get_float_ptr());
}

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

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx, Runtime *runtime) {
  OpTaskArgumentAccessor acc(task, regions, ctx, runtime);

  auto query = acc.get_tensor<READ_ONLY>(QUERY);
  auto key = acc.get_tensor<READ_ONLY>(KEY);
  auto value = acc.get_tensor<READ_ONLY>(VALUE);
  auto weight = acc.get_tensor<READ_ONLY>(WEIGHTS);
  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);

  auto output_grad = acc.get_tensor_grad<READ_ONLY>(OUTPUT);
  auto weight_grad = acc.get_tensor_grad<READ_WRITE>(WEIGHTS);
  auto query_grad = acc.get_tensor_grad<READ_WRITE>(QUERY);
  auto key_grad = acc.get_tensor_grad<READ_WRITE>(KEY);
  auto value_grad = acc.get_tensor_grad<READ_WRITE>(VALUE);

  float *key_grad_ptr =
      (key_grad == query_grad) ? nullptr : key_grad.get_float_ptr();
  float *value_grad_ptr = (value_grad == query_grad || value_grad == key_grad)
                              ? nullptr
                              : value_grad.get_float_ptr();

  assert(value_grad.shape == value.shape);
  assert(key_grad.shape == key.shape);

  assert(query_grad.shape == query.shape);
  assert(weight_grad.shape.get_volume() == weight.shape.get_volume());

  profile(backward_kernel, m->profiling,
          "[MultiHeadAttention] backward_time = %.2lfms\n", m,
          query.get_float_ptr(), query_grad.get_float_ptr(),
          key.get_float_ptr(), key_grad_ptr, value.get_float_ptr(),
          value_grad_ptr, weight.get_float_ptr(), weight_grad.get_float_ptr(),
          output_grad.get_float_ptr());
}

bool MultiHeadAttention::measure_operator_cost(
    Simulator *sim, MachineView const &mv, CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_query, sub_key, sub_value;
  if (!inputs[0]->get_sub_tensor(mv, sub_query)) {
    return false;
  }
  if (!inputs[1]->get_sub_tensor(mv, sub_key)) {
    return false;
  }
  if (!inputs[2]->get_sub_tensor(mv, sub_value)) {
    return false;
  }
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  // Currently assume only data parallel
  size_t num_weights = 0;
  {
    // Compute weight size
    int qSize = sub_query.dims[0].size;
    int kSize = sub_key.dims[0].size;
    int vSize = sub_value.dims[0].size;
    int qParas = qProjSize * qSize;
    int kParas = kProjSize(attrs) * kSize;
    int vParas = vProjSize(attrs) * vSize;
    int oParas =
        oProjSize(attrs) * (vProjSize(attrs) > 0 ? vProjSize(attrs) : vSize);
    num_weights = attrs.num_heads * (qParas + kParas + vParas + oParas);
  }
  assert(sub_query.num_dims == 4);
  int num_samples = sub_query.dims[2].size;

  auto allocator = std::unique_ptr<IAllocator>(new RealmAllocator(sim->memory));
  MHAPerDeviceState *m = new MHAPerDeviceState(
      sim->handler, this, allocator, attrs.num_samples, attrs.num_heads);

  // allocate tensors in simulator
  sim->free_all();
  float const *query_ptr =
      (float const *)sim->allocate(sub_query.get_volume(), DT_FLOAT);
  float const *key_ptr =
      (float const *)sim->allocate(sub_key.get_volume(), DT_FLOAT);
  float const *value_ptr =
      (float const *)sim->allocate(sub_value.get_volume(), DT_FLOAT);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float const *weight_ptr = (float const *)sim->allocate(num_weights, DT_FLOAT);
  cost_metrics.weights_memory += cost_metrics.total_mem_diff_from(sim->offset);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&](ffStream_t stream) {
    forward_kernel(m, query_ptr, key_ptr, value_ptr, weight_ptr, output_ptr);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *query_grad_ptr =
        (float *)sim->allocate(sub_query.get_volume(), DT_FLOAT);
    float *key_grad_ptr =
        (float *)sim->allocate(sub_key.get_volume(), DT_FLOAT);
    float *value_grad_ptr =
        (float *)sim->allocate(sub_value.get_volume(), DT_FLOAT);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *weight_grad_ptr = (float *)sim->allocate(num_weights, DT_FLOAT);
    cost_metrics.weights_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [&](ffStream_t stream) {
      backward_kernel(stream, m, query_ptr, query_grad_ptr, key_ptr,
                      key_grad_ptr, value_ptr, value_grad_ptr, weight_ptr,
                      weight_grad_ptr, output_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure MultiHeadAttention] query(%d %d %d) key(%d %d %d) "
           "value(%d %d %d) output(%d %d %d)"
           "forward_time(%.4lf) backward_time(%.4lf)\n",
           sub_query.dims[2].size, sub_query.dims[1].size,
           sub_query.dims[0].size, sub_key.dims[2].size, sub_key.dims[1].size,
           sub_key.dims[0].size, sub_value.dims[2].size, sub_value.dims[1].size,
           sub_value.dims[0].size, sub_output.dims[2].size,
           sub_output.dims[1].size, sub_output.dims[0].size,
           cost_metrics.forward_time, cost_metrics.backward_time);
  } else {
    printf("[Measure MultiHeadAttention] query(%d %d %d) key(%d %d %d) "
           "value(%d %d %d) output(%d %d %d)"
           "forward_time(%.4lf)\n",
           sub_query.dims[2].size, sub_query.dims[1].size,
           sub_query.dims[0].size, sub_key.dims[2].size, sub_key.dims[1].size,
           sub_key.dims[0].size, sub_value.dims[2].size, sub_value.dims[1].size,
           sub_value.dims[0].size, sub_output.dims[2].size,
           sub_output.dims[1].size, sub_output.dims[0].size,
           cost_metrics.forward_time);
  }
  // Free multiheadattentionmeta
  delete m;
  return true;
}

template <> void register_task<ATTENTION_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<MultiHeadAttentionAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);
  init.add_arg_slot<int>(KVSEQLENGTH);
  init.add_arg_slot<int>(KSIZE);
  init.add_arg_slot<int>(VSIZE);
  init.add_arg_slot<int>(QPROJSIZE);

  init.add_input_slot(QUERY);
  init.add_input_slot(KEY);
  init.add_input_slot(VALUE);
  init.add_param_slot(WEIGHTS);
  init.add_output_slot(OUTPUT);

  register_task(ATTENTION_INIT_TASK_ID, "MultiHeadAttention Init", init,
                init_task);
}

template <> void register_task<ATTENTION_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(QUERY);
  fwd.add_input_slot(KEY);
  fwd.add_input_slot(VALUE);
  fwd.add_param_slot(WEIGHTS);
  fwd.add_output_slot(OUTPUT);

  register_task(ATTENTION_FWD_TASK_ID, "MultiHeadAttention Fwd", fwd, fwd_task);
}

template <> void register_task<ATTENTION_BWD_TASK_ID>() {
  OpTaskSignature bwd(OpTaskType::BWD);

  bwd.add_input_slot(QUERY);
  bwd.add_input_grad_slot(QUERY);
  bwd.add_input_slot(KEY);
  bwd.add_input_grad_slot(KEY);
  bwd.add_input_slot(VALUE);
  bwd.add_input_slot(VALUE);

  bwd.add_param_slot(WEIGHTS);
  bwd.add_param_grad_slot(WEIGHTS);

  bwd.add_output_grad_slot(OUTPUT);

  register_task(ATTENTION_BWD_TASK_ID, "MultiHeadAttention Bwd", bwd, bwd_task);
}

} // namespace FlexFlow

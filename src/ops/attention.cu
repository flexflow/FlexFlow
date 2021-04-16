/* Copyright 2021 Facebook
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

#include "model.h"
#include "cuda_helper.h"

using namespace Legion;

Tensor FFModel::multihead_attention(const Tensor query,
                                    const Tensor key,
                                    const Tensor value,
                                    int embed_dim,
                                    int num_heads,
                                    int kdim,
                                    int vdim,
                                    float dropout,
                                    bool bias,
                                    bool add_bias_kv,
                                    bool add_zero_attn,
                                    Initializer* kernel_initializer,
                                    const char* name)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  //if (bias_initializer == NULL) {
  //  bias_initializer = new ZeroInitializer();
  //}
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
  Tensor kernel;
  {
    // Compute weight size
    int qSize = query->dims[0].size;
    int kSize = key->dims[0].size;
    int vSize = value->dims[0].size;
    int qProjSize = kdim;
    int kProjSize = kdim;
    int vProjSize = vdim;
    int oProjSize = embed_dim;
    int qParas = qProjSize * qSize;
    int kParas = kProjSize * kSize;
    int vParas = vProjSize * vSize;
    int oParas = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
    const int dims[2] = {num_heads, qParas + kParas + vParas + oParas};
    kernel = create_weight<2>(dims, DT_FLOAT, NULL/*owner_op*/,
        true/*create_grad*/, kernel_initializer, comm_type);
  }
  MultiHeadAttention* attn = new MultiHeadAttention(*this, query, key, value,
      kernel, embed_dim, num_heads, kdim, vdim, dropout, bias,
      add_bias_kv, add_zero_attn, name);
  layers.push_back(attn);
  return attn->outputs[0];
}

MultiHeadAttention::MultiHeadAttention(
    FFModel& model,
    const Tensor _query,
    const Tensor _key,
    const Tensor _value,
    const Tensor _weight,
    int _embed_dim, int _num_heads,
    int _kdim, int _vdim,
    float _dropout, bool _bias,
    bool _add_bias_kv, bool _add_zero_attn,
    const char* name)
//    Initializer* _bias_initializer)
: Op(model,
     OP_MULTIHEAD_ATTENTION,
     name, 3/*inputs*/, 1/*weights*/,
     _query, _key, _value, _weight),
  dropout(_dropout), bias(_bias),
  add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
  qSize(_query->dims[0].size), kSize(_key->dims[0].size), vSize(_value->dims[0].size),
  qProjSize(_kdim), kProjSize(_kdim), vProjSize(_vdim), oProjSize(_embed_dim),
  qoSeqLength(_query->dims[1].size), kvSeqLength(_key->dims[1].size)
  //bias_initializer(_bias_initializer)
{
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  numOutputs = 1;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < _query->num_dims; i++)
    dims[i] = _query->dims[_query->num_dims-1-i];
  dims[_query->num_dims-1].size = _embed_dim;
  outputs[0] = model.create_tensor(_query->num_dims, dims, DT_FLOAT, this);
}

#ifdef DEADCODE
void MultiHeadAttention::create_weights(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = model.get_or_create_task_is(3, pcname);
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
  {
    const int dims[2] = {weights[0].dims[1].size, weights[0].dims[0].size};
    weights[0] = model.create_linear_weight<2, 3>(this, dims, DT_FLOAT,
        kernel_initializer, true/*create_grad*/, comm_type);
  }
}
#endif

#ifdef DEADCODE
void MultiHeadAttention::create_input_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = model.get_or_create_task_is(3, pcname);

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<3> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_part_n = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_part_v = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_part_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  // Currently assume only partition over the batch dim
  assert(num_part_v == 1);
  assert(num_part_c == 1);
  return Op::create_input_partition(model);
  //{
  //  const int dims[3] = {outputs[0].dims[2].size, outputs[0].dims[1].size, outputs[0].dims[0].size};
  //  outputs[0] = model.create_tensor<3>(dims, DT_FLOAT, this);
  //  outputs[0].owner_op = this;
  //  outputs[0].owner_idx = 0;
  //}
  //for (int i = 0; i < 3; i++) {
  //  Rect<3> input_rect = runtime->get_index_partition_color_space(
  //      ctx, inputs[i]->part.get_index_partition());
  //  if (input_rect == part_rect) {
  //    input_lps[i] = inputs[i]->part;
  //    input_grad_lps[i] = inputs[i]->part_grad;
  //  } else {
  //    model.create_disjoint_partition(
  //        inputs[i], (IndexSpaceT<3>)task_is, input_lps[i], input_grad_lps[i]);
  //  }
  //}
}
#endif

/*
  regions[0](I): query
  regions[1](I): key
  regions[2](I): value
  regions[3](I): weight
  regions[4](O): output
*/
OpMeta* MultiHeadAttention::init_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  const MultiHeadAttention* attn = (MultiHeadAttention*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  TensorAccessorR<float, 3> acc_query(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_key(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_value(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 2> acc_weight(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 3> acc_output(
      regions[4], task->regions[4], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  int num_samples = acc_query.rect.hi[2] - acc_query.rect.lo[2] + 1;
  assert(attn->qoSeqLength == acc_query.rect.hi[1] - acc_query.rect.lo[1] + 1);
  assert(attn->qSize == acc_query.rect.hi[0]-acc_query.rect.lo[0]+1);
  assert(num_samples == acc_key.rect.hi[2]-acc_key.rect.lo[2]+1);
  assert(attn->kvSeqLength == acc_key.rect.hi[1]-acc_key.rect.lo[1]+1);
  assert(attn->kSize == acc_key.rect.hi[0]-acc_key.rect.lo[0]+1);
  assert(num_samples == acc_value.rect.hi[2]-acc_value.rect.lo[2]+1);
  assert(attn->kvSeqLength == acc_value.rect.hi[1]-acc_value.rect.lo[1]+1);
  assert(attn->vSize == acc_value.rect.hi[0]-acc_value.rect.lo[0]+1);
  int num_heads = acc_weight.rect.hi[1]-acc_weight.rect.lo[1]+1;
  assert(num_samples == acc_output.rect.hi[2]-acc_output.rect.lo[2]+1);
  assert(attn->qoSeqLength == acc_output.rect.hi[1]-acc_output.rect.lo[1]+1);
  assert(attn->oProjSize == acc_output.rect.hi[0]-acc_output.rect.lo[0]+1);

  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
         .only_kind(Memory::GPU_FB_MEM).best_affinity_to(task->target_proc).first();
  MultiHeadAttentionMeta* m = new MultiHeadAttentionMeta(handle,
      attn, gpu_mem, num_samples, num_heads);
  m->profiling = attn->profiling;
  assert(acc_weight.rect.volume() * sizeof(float) == m->weightSize);
  return m;
}

void MultiHeadAttention::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ATTENTION_INIT_TASK_ID, parallel_is,
      TaskArgument(this, sizeof(MultiHeadAttention)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[2]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
          WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(4, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*static*/
void MultiHeadAttention::forward_kernel(
    const MultiHeadAttentionMeta* m,
    const float* query_ptr,
    const float* key_ptr,
    const float* value_ptr,
    const float* weight_ptr,
    float* output_ptr)
{
  checkCUDNN(cudnnMultiHeadAttnForward(m->handle.dnn,
      m->attnDesc, -1, m->loWinIdx, m->hiWinIdx,
      m->devQoSeqArray, m->devKvSeqArray, m->qDesc,
      query_ptr, NULL/*residual*/, m->kDesc, key_ptr,
      m->vDesc, value_ptr, m->oDesc, output_ptr, m->weightSize,
      weight_ptr, m->handle.workSpaceSize, m->handle.workSpace,
      m->reserveSpaceSize, m->reserveSpace));
}

/*
  regions[0](I): query
  regions[1](I): key
  regions[2](I): value
  regions[3](I): weight
  regions[4](O): output
*/
__host__
void MultiHeadAttention::forward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  assert(regions.size() == 5);
  assert(task->regions.size() == regions.size());
  //const MultiHeadAttention* attn = (MultiHeadAttention*) task->args;
  const MultiHeadAttentionMeta* m = *((MultiHeadAttentionMeta**) task->local_args);
  TensorAccessorR<float, 3> acc_query(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_key(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_value(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 2> acc_weight(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 3> acc_output(
      regions[4], task->regions[4], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  MultiHeadAttention::forward_kernel(m,
      acc_query.ptr, acc_key.ptr, acc_value.ptr,
      acc_weight.ptr, acc_output.ptr);
  if (m->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("MultiHeadAttention forward time = %.2fms\n", elapsed);
    //print_tensor<3, float>(acc_query.ptr, acc_query.rect, "[Attention:forward:query]");
    //print_tensor<3, float>(acc_output.ptr, acc_output.rect, "[Attention:forward:output]");
  }
}

void MultiHeadAttention::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ATTENTION_FWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[2]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
          WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(4, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*static*/
void MultiHeadAttention::backward_kernel(
    const MultiHeadAttentionMeta* m,
    const float* query_ptr,
    float* query_grad_ptr,
    const float* key_ptr,
    float* key_grad_ptr,
    const float* value_ptr,
    float* value_grad_ptr,
    const float* weight_ptr,
    float* weight_grad_ptr,
    const float* output_grad_ptr)
{
  checkCUDNN(cudnnMultiHeadAttnBackwardData(m->handle.dnn,
      m->attnDesc, m->loWinIdx, m->hiWinIdx, m->devQoSeqArray,
      m->devKvSeqArray, m->oDesc, output_grad_ptr, m->qDesc,
      query_grad_ptr, query_ptr, m->kDesc, key_grad_ptr, key_ptr,
      m->vDesc, value_grad_ptr, value_ptr, m->weightSize, weight_ptr,
      m->handle.workSpaceSize, m->handle.workSpace, m->reserveSpaceSize,
      m->reserveSpace));
  checkCUDNN(cudnnMultiHeadAttnBackwardWeights(m->handle.dnn,
      m->attnDesc, CUDNN_WGRAD_MODE_ADD, m->qDesc,
      query_ptr, m->kDesc, key_ptr, m->vDesc, value_ptr, m->oDesc,
      output_grad_ptr, m->weightSize, weight_ptr, weight_grad_ptr,
      m->handle.workSpaceSize, m->handle.workSpace,
      m->reserveSpaceSize, m->reserveSpace));
}

/*
  regions[0](I): query
  regions[1](I): key
  regions[2](I): value
  regions[3](I): weight
  regions[4](I): output_grad
  regions[5](I/O): weight_grad
  regions[6](I/O): query_grad
  regions[7](I/O) (optional): key_grad
  regions[8](I/O) (optional): value_grad
*/
__host__
void MultiHeadAttention::backward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  assert(regions.size() >= 7);
  assert(task->regions.size() == regions.size());
  //MultiHeadAttention* attn = (MultiHeadAttention*) task->args;
  const MultiHeadAttentionMeta* m = *((MultiHeadAttentionMeta**) task->local_args);
  TensorAccessorR<float, 3> acc_query(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_key(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_value(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 2> acc_weight(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_output_grad(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_weight_grad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorW<float, 3> acc_query_grad(
      regions[6], task->regions[6], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  float *key_grad_ptr, *value_grad_ptr;
  assert(acc_query_grad.rect == acc_query.rect);
  assert(acc_weight_grad.rect.volume() == acc_weight.rect.volume());
  if (regions.size() == 7) {
    // assert query == key and query == value
    assert(regions[0].get_logical_region() == regions[1].get_logical_region());
    assert(regions[0].get_logical_region() == regions[2].get_logical_region());
    key_grad_ptr = acc_query_grad.ptr;
    value_grad_ptr = acc_query_grad.ptr;
  } else if (regions.size() == 8) {
    // assert query == key
    assert(regions[0].get_logical_region() == regions[1].get_logical_region());
    TensorAccessorW<float, 3> acc_value_grad(
        regions[7], task->regions[7], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    assert(acc_value_grad.rect == acc_value.rect);
    key_grad_ptr = acc_query_grad.ptr;
    value_grad_ptr = acc_value_grad.ptr;
  } else {
    assert(regions.size() == 10);
    TensorAccessorW<float, 3> acc_key_grad(
        regions[7], task->regions[7], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    TensorAccessorW<float, 3> acc_value_grad(
        regions[8], task->regions[8], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    assert(acc_key.rect == acc_key_grad.rect);
    assert(acc_value.rect == acc_value_grad.rect);
    value_grad_ptr = acc_value_grad.ptr;
    key_grad_ptr = acc_key_grad.ptr;
  }
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  MultiHeadAttention::backward_kernel(m,
      acc_query.ptr, acc_query_grad.ptr,
      acc_key.ptr, key_grad_ptr, acc_value.ptr, value_grad_ptr,
      acc_weight.ptr, acc_weight_grad.ptr,
      acc_output_grad.ptr);
  if (m->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("MultiHeadAttention backward time = %.2fms\n", elapsed);
  }
}

void MultiHeadAttention::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ATTENTION_BWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[2]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(4, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
          READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(5, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
          READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(6, FID_DATA);
  int num_regions = 7;
  if (inputs[1]->region != inputs[0]->region) {
    // when key != query
    launcher.add_region_requirement(
        RegionRequirement(inputs[1]->part_grad, 0/*projection id*/,
            READ_WRITE, EXCLUSIVE, inputs[1]->region_grad));
    launcher.add_field(num_regions++, FID_DATA);
  }
  if ((inputs[2]->region != inputs[0]->region)
  && (inputs[2]->region != inputs[1]->region)) {
    // when value != key and value != query
    launcher.add_region_requirement(
        RegionRequirement(inputs[2]->part_grad, 0/*projection id*/,
            READ_WRITE, EXCLUSIVE, inputs[2]->region_grad));
    launcher.add_field(num_regions++, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

MultiHeadAttentionMeta::MultiHeadAttentionMeta(FFHandler handler,
                                               const MultiHeadAttention* attn,
                                               Memory gpu_mem,
                                               int num_samples,
                                               int num_heads)
: OpMeta(handler)
{
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
#endif
  checkCUDNN(cudnnCreateAttnDescriptor(&attnDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&qDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&kDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&vDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&oDesc));
  // Currently do not support adding bias to key/value projection
  assert(!attn->add_bias_kv);
  cudnnAttnQueryMap_t attnMode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE;
  // Assume no beam search for now
  int maxBeamSize = 1;
  //printf("batchSize(%d) qSize(%d) kSize(%d) vSize(%d) qProjSize(%d) kProjSize(%d)\n",
  //    num_samples, attn->qSize, attn->kSize, attn->vSize, attn->qProjSize, attn->kProjSize);
  //printf("vProjSize(%d) oProjSize(%d) qoSeqLength(%d) kvSeqLength(%d)\n",
  //    attn->vProjSize, attn->oProjSize, attn->qoSeqLength, attn->kvSeqLength);
  cudnnMathType_t math_type;
  if (handle.allowTensorOpMathConversion) {
    math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  } else {
    math_type = CUDNN_TENSOR_OP_MATH;
  }
  checkCUDNN(cudnnSetAttnDescriptor(attnDesc, attnMode, num_heads,
      1.0f/*smScalar*/, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, math_type,
      NULL/*attnDropoutDesc*/, NULL/*postDropoutDesc*/,
      attn->qSize, attn->kSize, attn->vSize, attn->qProjSize, attn->kProjSize,
      attn->vProjSize, attn->oProjSize, attn->qoSeqLength, attn->kvSeqLength,
      num_samples, maxBeamSize));
  size_t workSpaceSize;
  checkCUDNN(cudnnGetMultiHeadAttnBuffers(handler.dnn, attnDesc, &weightSize,
      &workSpaceSize, &reserveSpaceSize));
  assert(workSpaceSize <= handler.workSpaceSize);
  //printf("weightSize(%zu) workSpaceSize(%zu) reserveSpaceSize(%zu)\n", weightSize, workSpaceSize, reserveSpaceSize);
  int dimA[CUDNN_SEQDATA_DIM_COUNT];
  cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
  assert(CUDNN_SEQDATA_DIM_COUNT == 4);
  axes[3] = CUDNN_SEQDATA_VECT_DIM; // 3 = nbDims-1
  axes[2] = CUDNN_SEQDATA_BEAM_DIM;
  axes[1] = CUDNN_SEQDATA_TIME_DIM;
  axes[0] = CUDNN_SEQDATA_BATCH_DIM;
  int *qoSeqArray = (int*) malloc(sizeof(int) * num_samples);
  int *kvSeqArray = (int*) malloc(sizeof(int) * num_samples);
  for (int i = 0; i < num_samples; i++) {
    qoSeqArray[i] = attn->qoSeqLength;
    kvSeqArray[i] = attn->kvSeqLength;
  }
  // Set qDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = attn->qoSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = attn->qSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(qDesc,
        CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
        num_samples, qoSeqArray, NULL));
  }
  // Set kDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = attn->kvSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = attn->kSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(kDesc,
        CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
        num_samples, kvSeqArray, NULL));
  }
  // Set vDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = attn->kvSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = attn->vSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(vDesc,
        CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
        num_samples, kvSeqArray, NULL));
  }
  // Set oDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = attn->qoSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = attn->oProjSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(oDesc,
        CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
        num_samples, qoSeqArray, NULL));
  }
  // allocate memory for the seqArray and reserve space
  {
    size_t totalSize = reserveSpaceSize + sizeof(int) * num_samples * 2;
    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0), Realm::Point<1, coord_t>(totalSize-1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(reserveInst, gpu_mem, bounds,
        field_sizes, 0, Realm::ProfilingRequestSet()).wait();
    devQoSeqArray = (int*) reserveInst.pointer_untyped(0, sizeof(char));
    checkCUDA(cudaMemcpy(devQoSeqArray, qoSeqArray, sizeof(int) * num_samples,
        cudaMemcpyHostToDevice));
    devKvSeqArray = (int*)devQoSeqArray + num_samples;
    checkCUDA(cudaMemcpy(devKvSeqArray, kvSeqArray, sizeof(int) * num_samples,
        cudaMemcpyHostToDevice));
    reserveSpace = (int*)devKvSeqArray + num_samples;
  }
  // allocate memory for loWinIdx/hiWinIdx
  loWinIdx = (int*) malloc(sizeof(int) * attn->qoSeqLength);
  hiWinIdx = (int*) malloc(sizeof(int) * attn->qoSeqLength);
  for (int i = 0; i < attn->qoSeqLength; i++) {
    loWinIdx[i] = 0;
    hiWinIdx[i] = attn->kvSeqLength;
  }
  free(qoSeqArray);
  free(kvSeqArray);
}

MultiHeadAttentionMeta::~MultiHeadAttentionMeta(void)
{
  reserveInst.destroy();
  free(loWinIdx);
  free(hiWinIdx);
  checkCUDNN(cudnnDestroyAttnDescriptor(attnDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(qDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(kDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(vDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(oDesc));
}

bool MultiHeadAttention::measure_operator_cost(Simulator* sim,
                                               const ParallelConfig& pc,
                                               CostMetrics& cost_metrics) const
{
  TensorBase sub_output, sub_query, sub_key, sub_value;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_query, OP_MULTIHEAD_ATTENTION))
    return false;
  if (!inputs[1]->get_input_sub_tensor(pc, sub_key, OP_MULTIHEAD_ATTENTION))
    return false;
  if (!inputs[2]->get_input_sub_tensor(pc, sub_value, OP_MULTIHEAD_ATTENTION))
    return false;
  if (!outputs[0]->get_input_sub_tensor(pc, sub_output, OP_MULTIHEAD_ATTENTION))
    return false;
  // Currently assume only data parallel
  Tensor sub_weight = weights[0];
  assert(sub_weight->num_dims == 2);
  int num_heads = sub_weight->dims[1].size;
  assert(sub_query.num_dims == 3);
  int num_samples = sub_query.dims[2].size;
  MultiHeadAttentionMeta* m = new MultiHeadAttentionMeta(sim->handler,
      this, sim->memory, num_samples, num_heads);

  // allocate tensors in simulator
  sim->free_all();
  const float* query_ptr =
      (const float*)sim->allocate(sub_query.get_volume(), DT_FLOAT);
  const float* key_ptr =
      (const float*)sim->allocate(sub_key.get_volume(), DT_FLOAT);
  const float* value_ptr =
      (const float*)sim->allocate(sub_value.get_volume(), DT_FLOAT);
  const float* weight_ptr =
      (const float*)sim->allocate(sub_weight->get_volume(), DT_FLOAT);
  float* output_ptr =
      (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, query_ptr, key_ptr, value_ptr, weight_ptr, output_ptr);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* query_grad_ptr =
        (float*)sim->allocate(sub_query.get_volume(), DT_FLOAT);
    float* key_grad_ptr =
        (float*)sim->allocate(sub_key.get_volume(), DT_FLOAT);
    float* value_grad_ptr =
        (float*)sim->allocate(sub_value.get_volume(), DT_FLOAT);
    float* weight_grad_ptr =
        (float*)sim->allocate(sub_weight->get_volume(), DT_FLOAT);
    float* output_grad_ptr =
        (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);

    backward = [&] {
      backward_kernel(m, query_ptr, query_grad_ptr, key_ptr, key_grad_ptr,
        value_ptr, value_grad_ptr, weight_ptr, weight_grad_ptr, output_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure MultiHeadAttention] query(%d %d %d) key(%d %d %d) value(%d %d %d) output(%d %d %d)"
         "forward_time(%.4lf) backward_time(%.4lf)\n",
         sub_query.dims[2].size, sub_query.dims[1].size, sub_query.dims[0].size,
         sub_key.dims[2].size, sub_key.dims[1].size, sub_key.dims[0].size,
         sub_value.dims[2].size, sub_value.dims[1].size, sub_value.dims[0].size,
         sub_output.dims[2].size, sub_output.dims[1].size, sub_output.dims[0].size,
         cost_metrics.forward_time, cost_metrics.backward_time);
  } else {
    printf("[Measure MultiHeadAttention] query(%d %d %d) key(%d %d %d) value(%d %d %d) output(%d %d %d)"
         "forward_time(%.4lf)\n",
         sub_query.dims[2].size, sub_query.dims[1].size, sub_query.dims[0].size,
         sub_key.dims[2].size, sub_key.dims[1].size, sub_key.dims[0].size,
         sub_value.dims[2].size, sub_value.dims[1].size, sub_value.dims[0].size,
         sub_output.dims[2].size, sub_output.dims[1].size, sub_output.dims[0].size,
         cost_metrics.forward_time);
  }
  // Free multiheadattentionmeta
  delete m;
  return true;
}

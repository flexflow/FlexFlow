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

#include "flexflow/ops/attention.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

/*static*/
void MultiHeadAttention::forward_kernel(MultiHeadAttentionMeta const *m,
                                        float const *query_ptr,
                                        float const *key_ptr,
                                        float const *value_ptr,
                                        float const *weight_ptr,
                                        float *output_ptr,
                                        hipStream_t stream) {
#if 0
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnMultiHeadAttnForward(m->handle.dnn,
                                       m->attnDesc, -1, m->loWinIdx, m->hiWinIdx,
                                       m->devQoSeqArray, m->devKvSeqArray, m->qDesc,
                                       query_ptr, NULL/*residual*/, m->kDesc, key_ptr,
                                       m->vDesc, value_ptr, m->oDesc, output_ptr, m->weightSize,
                                       weight_ptr, m->handle.workSpaceSize, m->handle.workSpace,
                                       m->reserveSpaceSize, m->reserveSpace));
#endif
}

/*static*/
void MultiHeadAttention::forward_kernel_wrapper(MultiHeadAttentionMeta const *m,
                                                float const *query_ptr,
                                                float const *key_ptr,
                                                float const *value_ptr,
                                                float const *weight_ptr,
                                                float *output_ptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  MultiHeadAttention::forward_kernel(
      m, query_ptr, key_ptr, value_ptr, weight_ptr, output_ptr, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("MultiHeadAttention forward time = %.2fms\n", elapsed);
    // print_tensor<3, float>(acc_query.ptr, acc_query.rect,
    // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
    // acc_output.rect, "[Attention:forward:output]");
  }
}

/*static*/
void MultiHeadAttention::backward_kernel(MultiHeadAttentionMeta const *m,
                                         float const *query_ptr,
                                         float *query_grad_ptr,
                                         float const *key_ptr,
                                         float *key_grad_ptr,
                                         float const *value_ptr,
                                         float *value_grad_ptr,
                                         float const *weight_ptr,
                                         float *weight_grad_ptr,
                                         float const *output_grad_ptr,
                                         hipStream_t stream) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

#if 0
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
#endif
}

/*static*/
void MultiHeadAttention::backward_kernel_wrapper(
    MultiHeadAttentionMeta const *m,
    float const *query_ptr,
    float *query_grad_ptr,
    float const *key_ptr,
    float *key_grad_ptr,
    float const *value_ptr,
    float *value_grad_ptr,
    float const *weight_ptr,
    float *weight_grad_ptr,
    float const *output_grad_ptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  MultiHeadAttention::backward_kernel(m,
                                      query_ptr,
                                      query_grad_ptr,
                                      key_ptr,
                                      key_grad_ptr,
                                      value_ptr,
                                      value_grad_ptr,
                                      weight_ptr,
                                      weight_grad_ptr,
                                      output_grad_ptr,
                                      stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("MultiHeadAttention backward time = %.2fms\n", elapsed);
  }
}

MultiHeadAttentionMeta::MultiHeadAttentionMeta(FFHandler handler,
                                               MultiHeadAttention const *attn,
                                               Memory gpu_mem,
                                               int num_samples,
                                               int num_heads)
    : OpMeta(handler) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(miopenSetStream(handler.dnn, stream));

#if 0
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
  hipdnnMathType_t math_type;
  if (handle.allowTensorOpMathConversion) {
    math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  } else {
    math_type = HIPDNN_TENSOR_OP_MATH;
  }
  checkCUDNN(cudnnSetAttnDescriptor(attnDesc, attnMode, num_heads,
      1.0f/*smScalar*/, HIPDNN_DATA_FLOAT, HIPDNN_DATA_FLOAT, math_type,
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
        HIPDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
        num_samples, qoSeqArray, NULL));
  }
  // Set kDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = attn->kvSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = attn->kSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(kDesc,
        HIPDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
        num_samples, kvSeqArray, NULL));
  }
  // Set vDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = attn->kvSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = attn->vSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(vDesc,
        HIPDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
        num_samples, kvSeqArray, NULL));
  }
  // Set oDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = attn->qoSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = attn->oProjSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(oDesc,
        HIPDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, axes,
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
    checkCUDA(hipMemcpy(devQoSeqArray, qoSeqArray, sizeof(int) * num_samples,
        hipMemcpyHostToDevice));
    devKvSeqArray = (int*)devQoSeqArray + num_samples;
    checkCUDA(hipMemcpy(devKvSeqArray, kvSeqArray, sizeof(int) * num_samples,
        hipMemcpyHostToDevice));
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
#endif
}

MultiHeadAttentionMeta::~MultiHeadAttentionMeta(void) {
#if 0
  reserveInst.destroy();
  free(loWinIdx);
  free(hiWinIdx);
  checkCUDNN(cudnnDestroyAttnDescriptor(attnDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(qDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(kDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(vDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(oDesc));
#endif
}

}; // namespace FlexFlow

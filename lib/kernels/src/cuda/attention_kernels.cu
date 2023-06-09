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

#include "kernels/attention_kernels.h"
#include "kernels/cuda_helper.h"

namespace FlexFlow {
namespace Kernels {
namespace MultiHeadAttention {

void init_kernel(MHAPerDeviceState *m,
                 int num_samples,
                 int num_heads,
                 int qSize,
                 int kSize,
                 int vSize,
                 int qProjSize,
                 int kProjSize,
                 int vProjSize,
                 int oProjSize,
                 int qoSeqLength,
                 int kvSeqLength,
                 bool add_bias_kv) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  checkCUDNN(cudnnCreateAttnDescriptor(&m->attnDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&m->qDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&m->kDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&m->vDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&m->oDesc));
  // Currently do not support adding bias to key/value projection
  assert(!add_bias_kv);
  cudnnAttnQueryMap_t attnMode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE;
  // Assume no beam search for now
  int maxBeamSize = 1;
  // printf("batchSize(%d) qSize(%d) kSize(%d) vSize(%d) qProjSize(%d)
  // kProjSize(%d)\n",
  //     num_samples, attn->qSize, attn->kSize, attn->vSize, attn->qProjSize,
  //     attn->kProjSize);
  // printf("vProjSize(%d) oProjSize(%d) qoSeqLength(%d) kvSeqLength(%d)\n",
  //     attn->vProjSize, attn->oProjSize, attn->qoSeqLength,
  //     attn->kvSeqLength);
  cudnnMathType_t math_type;
  if (m->handle.allowTensorOpMathConversion) {
    math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  } else {
    math_type = CUDNN_TENSOR_OP_MATH;
  }
  checkCUDNN(cudnnSetAttnDescriptor(m->attnDesc,
                                    attnMode,
                                    num_heads,
                                    1.0f /*smScalar*/,
                                    CUDNN_DATA_FLOAT,
                                    CUDNN_DATA_FLOAT,
                                    math_type,
                                    NULL /*attnDropoutDesc*/,
                                    NULL /*postDropoutDesc*/,
                                    qSize,
                                    kSize,
                                    vSize,
                                    qProjSize,
                                    kProjSize,
                                    vProjSize,
                                    oProjSize,
                                    qoSeqLength,
                                    kvSeqLength,
                                    num_samples,
                                    maxBeamSize));
  size_t workSpaceSize;
  checkCUDNN(cudnnGetMultiHeadAttnBuffers(m->handle.dnn,
                                          m->attnDesc,
                                          &m->weightSize,
                                          &workSpaceSize,
                                          &m->reserveSpaceSize));
  assert(workSpaceSize <= m->handle.workSpaceSize);
  // printf("weightSize(%zu) workSpaceSize(%zu) reserveSpaceSize(%zu)\n",
  // weightSize, workSpaceSize, reserveSpaceSize);
  int dimA[CUDNN_SEQDATA_DIM_COUNT];
  cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
  assert(CUDNN_SEQDATA_DIM_COUNT == 4);
  axes[3] = CUDNN_SEQDATA_VECT_DIM; // 3 = nbDims-1
  axes[2] = CUDNN_SEQDATA_BEAM_DIM;
  axes[1] = CUDNN_SEQDATA_TIME_DIM;
  axes[0] = CUDNN_SEQDATA_BATCH_DIM;
  int *qoSeqArray = (int *)malloc(sizeof(int) * num_samples);
  int *kvSeqArray = (int *)malloc(sizeof(int) * num_samples);
  for (int i = 0; i < num_samples; i++) {
    qoSeqArray[i] = qoSeqLength;
    kvSeqArray[i] = kvSeqLength;
  }
  // Set qDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = qoSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = qSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(m->qDesc,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_SEQDATA_DIM_COUNT,
                                         dimA,
                                         axes,
                                         num_samples,
                                         qoSeqArray,
                                         NULL));
  }
  // Set kDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = kvSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = kSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(m->kDesc,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_SEQDATA_DIM_COUNT,
                                         dimA,
                                         axes,
                                         num_samples,
                                         kvSeqArray,
                                         NULL));
  }
  // Set vDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = kvSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = vSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(m->vDesc,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_SEQDATA_DIM_COUNT,
                                         dimA,
                                         axes,
                                         num_samples,
                                         kvSeqArray,
                                         NULL));
  }
  // Set oDesc
  {
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[CUDNN_SEQDATA_TIME_DIM] = qoSeqLength;
    dimA[CUDNN_SEQDATA_VECT_DIM] = oProjSize;
    checkCUDNN(cudnnSetSeqDataDescriptor(m->oDesc,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_SEQDATA_DIM_COUNT,
                                         dimA,
                                         axes,
                                         num_samples,
                                         qoSeqArray,
                                         NULL));
  }
  // allocate memory for the seqArray and reserve space
  {
    size_t totalSize = m->reserveSpaceSize + sizeof(int) * num_samples * 2;

    m->devQoSeqArray = (int *)m->gpu_alloc(totalSize);
    checkCUDA(cudaMemcpy(m->devQoSeqArray,
                         qoSeqArray,
                         sizeof(int) * num_samples,
                         cudaMemcpyHostToDevice));
    m->devKvSeqArray = m->devQoSeqArray + num_samples;
    checkCUDA(cudaMemcpy(m->devKvSeqArray,
                         kvSeqArray,
                         sizeof(int) * num_samples,
                         cudaMemcpyHostToDevice));
    m->reserveSpace = m->devKvSeqArray + num_samples;
  }
  // allocate memory for loWinIdx/hiWinIdx
  m->loWinIdx = (int *)malloc(sizeof(int) * qoSeqLength);
  m->hiWinIdx = (int *)malloc(sizeof(int) * qoSeqLength);
  for (int i = 0; i < qoSeqLength; i++) {
    m->loWinIdx[i] = 0;
    m->hiWinIdx[i] = kvSeqLength;
  }
  free(qoSeqArray);
  free(kvSeqArray);
}

/* void forward_kernel_wrapper(MHAPerDeviceState const *m, */
/*                                                 float const *query_ptr, */
/*                                                 float const *key_ptr, */
/*                                                 float const *value_ptr, */
/*                                                 float const *weight_ptr, */
/*                                                 float *output_ptr) { */
/*   wrapper(Internal::forward_kernel, m->profiling, ) */
/*   cudaStream_t stream; */
/*   checkCUDA(get_legion_stream(&stream)); */

/*   cudaEvent_t t_start, t_end; */
/*   if (m->profiling) { */
/*     cudaEventCreate(&t_start); */
/*     cudaEventCreate(&t_end); */
/*     cudaEventRecord(t_start, stream); */
/*   } */
/*   Internal::forward_kernel( */
/*       m, query_ptr, key_ptr, value_ptr, weight_ptr, output_ptr, stream); */
/*   if (m->profiling) { */
/*     cudaEventRecord(t_end, stream); */
/*     checkCUDA(cudaEventSynchronize(t_end)); */
/*     float elapsed = 0; */
/*     checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end)); */
/*     cudaEventDestroy(t_start); */
/*     cudaEventDestroy(t_end); */
/*     printf("MultiHeadAttention forward time = %.2fms\n", elapsed); */
/*     // print_tensor<3, float>(acc_query.ptr, acc_query.rect, */
/*     // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
 */
/*     // acc_output.rect, "[Attention:forward:output]"); */
/*   } */
/* } */

/* void backward_kernel_wrapper( */
/*     MHAPerDeviceState const *m, */
/*     float const *query_ptr, */
/*     float *query_grad_ptr, */
/*     float const *key_ptr, */
/*     float *key_grad_ptr, */
/*     float const *value_ptr, */
/*     float *value_grad_ptr, */
/*     float const *weight_ptr, */
/*     float *weight_grad_ptr, */
/*     float const *output_grad_ptr) { */
/*   cudaStream_t stream; */
/*   checkCUDA(get_legion_stream(&stream)); */

/*   cudaEvent_t t_start, t_end; */
/*   if (m->profiling) { */
/*     cudaEventCreate(&t_start); */
/*     cudaEventCreate(&t_end); */
/*     cudaEventRecord(t_start, stream); */
/*   } */

/*   Internal::backward_kernel(m, */
/*                                       query_ptr, */
/*                                       query_grad_ptr, */
/*                                       key_ptr, */
/*                                       key_grad_ptr, */
/*                                       value_ptr, */
/*                                       value_grad_ptr, */
/*                                       weight_ptr, */
/*                                       weight_grad_ptr, */
/*                                       output_grad_ptr, */
/*                                       stream); */
/*   if (m->profiling) { */
/*     cudaEventRecord(t_end, stream); */
/*     checkCUDA(cudaEventSynchronize(t_end)); */
/*     float elapsed = 0; */
/*     checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end)); */
/*     cudaEventDestroy(t_start); */
/*     cudaEventDestroy(t_end); */
/*     printf("MultiHeadAttention backward time = %.2fms\n", elapsed); */
/*   } */
/* } */

/* namespace Internal { */

void forward_kernel(cudaStream_t stream,
                    MHAPerDeviceState *m,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnMultiHeadAttnForward(m->handle.dnn,
                                       m->attnDesc,
                                       -1,
                                       m->loWinIdx,
                                       m->hiWinIdx,
                                       m->devQoSeqArray,
                                       m->devKvSeqArray,
                                       m->qDesc,
                                       query_ptr,
                                       nullptr /*residual*/,
                                       m->kDesc,
                                       key_ptr,
                                       m->vDesc,
                                       value_ptr,
                                       m->oDesc,
                                       output_ptr,
                                       m->weightSize,
                                       weight_ptr,
                                       m->handle.workSpaceSize,
                                       m->handle.workSpace,
                                       m->reserveSpaceSize,
                                       m->reserveSpace));
}

void backward_kernel(cudaStream_t stream,
                     MHAPerDeviceState *m,
                     float const *query_ptr,
                     float *query_grad_ptr,
                     float const *key_ptr,
                     float *key_grad_ptr,
                     float const *value_ptr,
                     float *value_grad_ptr,
                     float const *weight_ptr,
                     float *weight_grad_ptr,
                     float const *output_grad_ptr) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnMultiHeadAttnBackwardData(m->handle.dnn,
                                            m->attnDesc,
                                            m->loWinIdx,
                                            m->hiWinIdx,
                                            m->devQoSeqArray,
                                            m->devKvSeqArray,
                                            m->oDesc,
                                            output_grad_ptr,
                                            m->qDesc,
                                            query_grad_ptr,
                                            query_ptr,
                                            m->kDesc,
                                            key_grad_ptr,
                                            key_ptr,
                                            m->vDesc,
                                            value_grad_ptr,
                                            value_ptr,
                                            m->weightSize,
                                            weight_ptr,
                                            m->handle.workSpaceSize,
                                            m->handle.workSpace,
                                            m->reserveSpaceSize,
                                            m->reserveSpace));
  checkCUDNN(cudnnMultiHeadAttnBackwardWeights(m->handle.dnn,
                                               m->attnDesc,
                                               CUDNN_WGRAD_MODE_ADD,
                                               m->qDesc,
                                               query_ptr,
                                               m->kDesc,
                                               key_ptr,
                                               m->vDesc,
                                               value_ptr,
                                               m->oDesc,
                                               output_grad_ptr,
                                               m->weightSize,
                                               weight_ptr,
                                               weight_grad_ptr,
                                               m->handle.workSpaceSize,
                                               m->handle.workSpace,
                                               m->reserveSpaceSize,
                                               m->reserveSpace));
}

/* } // namespace Internal */
} // namespace MultiHeadAttention
} // namespace Kernels

MHAPerDeviceState::MHAPerDeviceState(FFHandler handler,
                                     Memory gpu_mem,
                                     int num_samples,
                                     int num_heads,
                                     int qSize,
                                     int kSize,
                                     int vSize,
                                     int qProjSize,
                                     int kProjSize,
                                     int vProjSize,
                                     int oProjSize,
                                     int qoSeqLength,
                                     int kvSeqLength,
                                     bool add_bias_kv)
    : PerDeviceOpState(handler) {}

MHAPerDeviceState::MHAPerDeviceState(FFHandler handler,
                                     std::unique_ptr<IAllocator> allocator,
                                     MultiHeadAttentionAttrs const &attrs,
                                     ArrayShape const &query_shape,
                                     ArrayShape const &key_shape,
                                     ArrayShape const &value_shape) {
  : MHAPerDeviceState(handler, 
                      allocator, 
                      query_shape[2],
                      attrs.num_heads, 
                      query_shape[0],
                      key_shape[0],
                      value_shape[0],
                      qProjSize(attrs),
                      kProjSize(attrs),
                      vProjSize(attrs),
                      oProjSize(attrs),
                      query_shape[1],
                      key_shape[1],
                      attrs.add_bias_kv)
{ }

  MHAPerDeviceState::~MHAPerDeviceState(void) {
    free(loWinIdx);
    free(hiWinIdx);
    checkCUDNN(cudnnDestroyAttnDescriptor(attnDesc));
    checkCUDNN(cudnnDestroySeqDataDescriptor(qDesc));
    checkCUDNN(cudnnDestroySeqDataDescriptor(kDesc));
    checkCUDNN(cudnnDestroySeqDataDescriptor(vDesc));
    checkCUDNN(cudnnDestroySeqDataDescriptor(oDesc));
  }

} // namespace FlexFlow

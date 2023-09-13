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
#include "device.h"
#include "kernels/device.h"

namespace FlexFlow {
namespace Kernels {
namespace MultiHeadAttention {

MHAPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                              Allocator allocator,
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
  ffAttnDescriptor_t attnDesc;
  ffSeqDataDescriptor_t qDesc;
  ffSeqDataDescriptor_t kDesc;
  ffSeqDataDescriptor_t vDesc;
  ffSeqDataDescriptor_t oDesc;
  void *reserveSpace;
  void *dropoutStates;
  int *devQoSeqArray;
  int *devKvSeqArray;
  size_t reserveSpaceSize;
  size_t dropoutStateSize;
  size_t weightSize;

  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handle.dnn, stream));
  checkCUDNN(cudnnCreateAttnDescriptor(&attnDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&qDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&kDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&vDesc));
  checkCUDNN(cudnnCreateSeqDataDescriptor(&oDesc));

  // Currently do not support adding bias to key/value projection
  assert(!add_bias_kv);
  cudnnAttnQueryMap_t attnMode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE;

  // Assume no beam search for now
  int maxBeamSize = 1;

  cudnnMathType_t math_type;
  if (handle.allowTensorOpMathConversion) {
    math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  } else {
    math_type = CUDNN_TENSOR_OP_MATH;
  }
  checkCUDNN(cudnnSetAttnDescriptor(attnDesc,
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
  checkCUDNN(cudnnGetMultiHeadAttnBuffers(handle.dnn,
                                          attnDesc,
                                          &weightSize,
                                          &workSpaceSize,
                                          &reserveSpaceSize));
  assert(workSpaceSize <= handle.workSpaceSize);

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
    checkCUDNN(cudnnSetSeqDataDescriptor(qDesc,
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
    checkCUDNN(cudnnSetSeqDataDescriptor(kDesc,
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
    checkCUDNN(cudnnSetSeqDataDescriptor(vDesc,
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
    checkCUDNN(cudnnSetSeqDataDescriptor(oDesc,
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
    size_t totalSize = reserveSpaceSize + sizeof(int) * num_samples * 2;

    devQoSeqArray = (int *)allocator.allocate(totalSize);
    checkCUDA(cudaMemcpy(devQoSeqArray,
                         qoSeqArray,
                         sizeof(int) * num_samples,
                         cudaMemcpyHostToDevice));
    devKvSeqArray = devQoSeqArray + num_samples;
    checkCUDA(cudaMemcpy(devKvSeqArray,
                         kvSeqArray,
                         sizeof(int) * num_samples,
                         cudaMemcpyHostToDevice));
    reserveSpace = devKvSeqArray + num_samples;
  }
  // allocate memory for loWinIdx/hiWinIdx
  int *loWinIdx = (int *)malloc(sizeof(int) * qoSeqLength);
  int *hiWinIdx = (int *)malloc(sizeof(int) * qoSeqLength);
  for (int i = 0; i < qoSeqLength; i++) {
    loWinIdx[i] = 0;
    hiWinIdx[i] = kvSeqLength;
  }

  MHAPerDeviceState per_device_state = {handle,
                                        weightSize,
                                        reserveSpaceSize,
                                        attnDesc,
                                        qDesc,
                                        kDesc,
                                        vDesc,
                                        oDesc,
                                        devQoSeqArray,
                                        devKvSeqArray,
                                        loWinIdx,
                                        hiWinIdx,
                                        reserveSpace,
                                        allocator};
  free(qoSeqArray);
  free(kvSeqArray);
}

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

void cleanup_kernel(int *loWinIdx,
                    int *hiWinIdx,
                    ffAttnDescriptor_t attnDesc,
                    ffSeqDataDescriptor_t qDesc,
                    ffSeqDataDescriptor_t kDesc,
                    ffSeqDataDescriptor_t vDesc,
                    ffSeqDataDescriptor_t oDesc) {
  free(loWinIdx);
  free(hiWinIdx);
  checkCUDNN(cudnnDestroyAttnDescriptor(attnDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(qDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(kDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(vDesc));
  checkCUDNN(cudnnDestroySeqDataDescriptor(oDesc));
}

} // namespace MultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

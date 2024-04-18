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
#include <hip/hip_runtime.h>

namespace FlexFlow {
namespace Kernels {
namespace MultiHeadAttention {

MHAPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                              Allocator &allocator,
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
  hipStream_t stream;
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
  checkCUDNN(miopenSetStream(handler.dnn, stream));
  checkCUDNN(miopenCreateAttnDescriptor(&attnDesc));
  checkCUDNN(miopenCreateSeqDataDescriptor(&qDesc));
  checkCUDNN(miopenCreateSeqDataDescriptor(&kDesc));
  checkCUDNN(miopenCreateSeqDataDescriptor(&vDesc));
  checkCUDNN(miopenCreateSeqDataDescriptor(&oDesc));

  assert(!add_bias_kv);
  miopenAttnQueryMap_t attnMode = MIOPEN_ATTN_QUERYMAP_ALL_TO_ONE;

  int maxBeamSize = 1;

  hipdnnMathType_t math_type;
  if (handle.allowTensorOpMathConversion) {
    math_type = HIPDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  } else {
    math_type = HIPDNN_TENSOR_OP_MATH;
  }
  checkCUDNN(miopenSetAttnDescriptor(attnDesc,
                                     attnMode,
                                     num_heads,
                                     1.0f /*smScalar*/,
                                     HIPDNN_DATA_FLOAT,
                                     HIPDNN_DATA_FLOAT,
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
  checkCUDNN(miopenGetMultiHeadAttnBuffers(
      handler.dnn, attnDesc, &weightSize, &workSpaceSize, &reserveSpaceSize));
  assert(workSpaceSize <= handler.workSpaceSize);

  int dimA[MIOPEN_SEQDATA_DIM_COUNT];
  miopenSeqDataAxis_t axes[MIOPEN_SEQDATA_DIM_COUNT];
  assert(MIOPEN_SEQDATA_DIM_COUNT == 4);
  axes[3] = MIOPEN_SEQDATA_VECT_DIM; // 3 = nbDims-1
  axes[2] = MIOPEN_SEQDATA_BEAM_DIM;
  axes[1] = MIOPEN_SEQDATA_TIME_DIM;
  axes[0] = MIOPEN_SEQDATA_BATCH_DIM;

  std::unique_ptr<int[]> qoSeqArray(new int[num_samples]);
  std::unique_ptr<int[]> kvSeqArray(new int[num_samples]);
  for (int i = 0; i < num_samples; i++) {
    qoSeqArray[i] = qoSeqLength;
    kvSeqArray[i] = kvSeqLength;
  }

  // Set qDesc
  {
    dimA[MIOPEN_SEQDATA_BEAM_DIM] = 1;
    dimA[MIOPEN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[MIOPEN_SEQDATA_TIME_DIM] = qoSeqLength;
    dimA[MIOPEN_SEQDATA_VECT_DIM] = qSize;
    checkCUDNN(miopenSetSeqDataDescriptor(qDesc,
                                          MIOPEN_DATA_FLOAT,
                                          MIOPEN_SEQDATA_DIM_COUNT,
                                          dimA,
                                          axes,
                                          num_samples,
                                          qoSeqArray.get(),
                                          NULL));
  }
  // Set kDesc
  {
    dimA[MIOPEN_SEQDATA_BEAM_DIM] = 1;
    dimA[MIOPEN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[MIOPEN_SEQDATA_TIME_DIM] = kvSeqLength;
    dimA[MIOPEN_SEQDATA_VECT_DIM] = kSize;
    checkCUDNN(miopenSetSeqDataDescriptor(kDesc,
                                          MIOPEN_DATA_FLOAT,
                                          MIOPEN_SEQDATA_DIM_COUNT,
                                          dimA,
                                          axes,
                                          num_samples,
                                          kvSeqArray.get(),
                                          NULL));
  }
  // Set vDesc
  {
    dimA[MIOPEN_SEQDATA_BEAM_DIM] = 1;
    dimA[MIOPEN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[MIOPEN_SEQDATA_TIME_DIM] = kvSeqLength;
    dimA[MIOPEN_SEQDATA_VECT_DIM] = vSize;
    checkCUDNN(miopenSetSeqDataDescriptor(vDesc,
                                          MIOPEN_DATA_FLOAT,
                                          MIOPEN_SEQDATA_DIM_COUNT,
                                          dimA,
                                          axes,
                                          num_samples,
                                          kvSeqArray.get(),
                                          NULL));
  }
  // Set oDesc
  {
    dimA[MIOPEN_SEQDATA_BEAM_DIM] = 1;
    dimA[MIOPEN_SEQDATA_BATCH_DIM] = num_samples;
    dimA[MIOPEN_SEQDATA_TIME_DIM] = qoSeqLength;
    dimA[MIOPEN_SEQDATA_VECT_DIM] = oProjSize;
    checkCUDNN(miopenSetSeqDataDescriptor(oDesc,
                                          MIOPEN_DATA_FLOAT,
                                          MIOPEN_SEQDATA_DIM_COUNT,
                                          dimA,
                                          axes,
                                          num_samples,
                                          qoSeqArray.get(),
                                          NULL));
  }

  // allocate memory for the seqArray and reserve space
  {
    size_t totalSize = reserveSpaceSize + sizeof(int) * num_samples * 2;

    devQoSeqArray = (int *)allocator.allocate(totalSize);
    checkCUDA(miopenMemcpy(devQoSeqArray,
                           qoSeqArray.get(),
                           sizeof(int) * num_samples,
                           miopenMemcpyHostToDevice));
    devKvSeqArray = devQoSeqArray + num_samples;
    checkCUDA(miopenMemcpy(devKvSeqArray,
                           kvSeqArray.get(),
                           sizeof(int) * num_samples,
                           miopenMemcpyHostToDevice));
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

  return per_device_state;
}

void forward_kernel(hipStream_t stream,
                    MHAPerDeviceState const &device_state,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr) {

  checkCUDNN(miopenSetStream(device_state.handle.dnn, stream));

  checkCUDNN(miopenMultiHeadAttnForward(device_state.handle.dnn,
                                        device_state.attnDesc,
                                        device_state.loWinIdx,
                                        device_state.hiWinIdx,
                                        device_state.devQoSeqArray,
                                        device_state.devKvSeqArray,
                                        device_state.oDesc,
                                        output_ptr,
                                        device_state.qDesc,
                                        query_ptr,
                                        device_state.kDesc,
                                        key_ptr,
                                        device_state.vDesc,
                                        value_ptr,
                                        weight_ptr,
                                        device_state.weightSize,
                                        device_state.reserveSpaceSize,
                                        device_state.reserveSpace));
#endif
}

void backward_kernel(hipStream_t stream,
                     MHAPerDeviceState const &device_state,
                     float const *query_ptr,
                     float *query_grad_ptr,
                     float const *key_ptr,
                     float *key_grad_ptr,
                     float const *value_ptr,
                     float *value_grad_ptr,
                     float const *weight_ptr,
                     float *weight_grad_ptr,
                     float const *output_grad_ptr) {
  checkCUDNN(miopenSetStream(device_state.handle.dnn, stream));

  checkCUDNN(miopenMultiHeadAttnBackwardData(device_state.handle.dnn,
                                             device_state.attnDesc,
                                             device_state.loWinIdx,
                                             device_state.hiWinIdx,
                                             device_state.devQoSeqArray,
                                             device_state.devKvSeqArray,
                                             device_state.oDesc,
                                             output_grad_ptr,
                                             device_state.qDesc,
                                             query_grad_ptr,
                                             query_ptr,
                                             device_state.kDesc,
                                             key_grad_ptr,
                                             key_ptr,
                                             device_state.vDesc,
                                             value_grad_ptr,
                                             value_ptr,
                                             weight_ptr,
                                             device_state.weightSize,
                                             device_state.reserveSpaceSize,
                                             device_state.reserveSpace));

  checkCUDNN(miopenMultiHeadAttnBackwardWeights(device_state.handle.dnn,
                                                device_state.attnDesc,
                                                device_state.loWinIdx,
                                                device_state.hiWinIdx,
                                                device_state.devQoSeqArray,
                                                device_state.devKvSeqArray,
                                                device_state.oDesc,
                                                output_grad_ptr,
                                                device_state.qDesc,
                                                query_ptr,
                                                device_state.kDesc,
                                                key_ptr,
                                                device_state.vDesc,
                                                value_ptr,
                                                weight_grad_ptr,
                                                device_state.weightSize,
                                                device_state.reserveSpaceSize,
                                                device_state.reserveSpace));
}

void cleanup_kernel(Allocator &allocator,
                    MHAPerDeviceState const &device_state) {
  allocator.deallocate(device_state.loWinIdx);
  allocator.deallocate(device_state.hiWinIdx);
  checkCUDNN(miopenDestroyAttnDescriptor(device_state.attnDesc));
  checkCUDNN(miopenDestroySeqDataDescriptor(device_state.qDesc));
  checkCUDNN(miopenDestroySeqDataDescriptor(device_state.kDesc));
  checkCUDNN(miopenDestroySeqDataDescriptor(device_state.vDesc));
  checkCUDNN(miopenDestroySeqDataDescriptor(device_state.oDesc));
}

} // namespace MultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

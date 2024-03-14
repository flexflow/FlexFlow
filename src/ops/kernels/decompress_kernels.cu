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
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

namespace Kernels {

template <typename DT>
__global__ void decompress_int4_general_weights(char const *input_weight_ptr,
                                                DT *weight_ptr,
                                                int in_dim,
                                                int valueSize) {
  // eg. in dim = 3072, out dim = 768
  CUDA_KERNEL_LOOP(i, valueSize / 2) {
    size_t real_idx_first = i * 2;
    size_t real_idx_second = i * 2 + 1;
    size_t group_idx =
        (real_idx_first / (in_dim * INT4_NUM_OF_ELEMENTS_PER_GROUP)) * in_dim +
        real_idx_first % in_dim;
    size_t idx = i;
    size_t offset_idx = (valueSize / 2) + group_idx * sizeof(DT);
    size_t scale_idx = offset_idx + sizeof(DT) * (valueSize / 32);

    weight_ptr[real_idx_first] =
        static_cast<DT>((input_weight_ptr[idx] >> 4) & 0xF) /
            (*(DT *)(input_weight_ptr + scale_idx)) +
        (*(DT *)(input_weight_ptr + offset_idx));
    weight_ptr[real_idx_second] =
        static_cast<DT>(input_weight_ptr[idx] & 0xF) /
            (*(DT *)(input_weight_ptr + scale_idx + sizeof(DT))) +
        (*(DT *)(input_weight_ptr + offset_idx + sizeof(DT)));
  }
}

template <typename DT>
__global__ void decompress_int8_general_weights(char const *input_weight_ptr,
                                                DT *weight_ptr,
                                                int in_dim,
                                                int valueSize) {
  CUDA_KERNEL_LOOP(i, valueSize) {
    size_t idx = i;
    size_t group_idx =
        (idx / (in_dim * INT4_NUM_OF_ELEMENTS_PER_GROUP)) * in_dim +
        idx % in_dim;
    size_t offset_idx = valueSize + group_idx * sizeof(DT);
    size_t scale_idx = offset_idx + sizeof(DT) * (valueSize / 32);
    weight_ptr[idx] = static_cast<DT>(input_weight_ptr[idx] & 0xFF) /
                          (*(DT *)(input_weight_ptr + scale_idx)) +
                      (*(DT *)(input_weight_ptr + offset_idx));
  }
}

template <typename DT>
__global__ void decompress_int4_attention_weights(char *input_weight_ptr,
                                                  DT *weight_ptr,
                                                  int qProjSize,
                                                  int qSize,
                                                  int num_heads) {
  // TODO this is because in top level function we assume q,k,v in same size
  CUDA_KERNEL_LOOP(i, qProjSize * num_heads * qSize / 2) {
    int q_block_size = (qProjSize * qSize) / 2;
    int real_q_block_size = q_block_size * 2;
    size_t qkvo_block_size = q_block_size * 4;
    size_t real_qkvo_block_size = qkvo_block_size * 2;

    int group_idx = (i * 2 / (INT4_NUM_OF_ELEMENTS_PER_GROUP * qSize)) * qSize +
                    (i * 2) % qSize;
    // i * 2 / (INT4_NUM_OF_ELEMENTS_PER_GROUP);
    int head_idx = i / q_block_size;
    int data_idx = i % q_block_size;

    size_t idx_q = head_idx * qkvo_block_size + data_idx;
    size_t idx_k = idx_q + q_block_size;
    size_t idx_v = idx_k + q_block_size;
    size_t idx_o = idx_v + q_block_size;

    size_t real_idx_q_first = head_idx * real_qkvo_block_size + data_idx * 2;
    size_t real_idx_q_second = real_idx_q_first + 1;
    size_t real_idx_k_first =
        head_idx * real_qkvo_block_size + real_q_block_size + data_idx * 2;
    size_t real_idx_k_second = real_idx_k_first + 1;
    size_t real_idx_v_first =
        head_idx * real_qkvo_block_size + real_q_block_size * 2 + data_idx * 2;
    size_t real_idx_v_second = real_idx_v_first + 1;
    size_t real_idx_o_first =
        head_idx * real_qkvo_block_size + real_q_block_size * 3 + data_idx * 2;
    size_t real_idx_o_second = real_idx_o_first + 1;

    size_t meta_offset = num_heads * qkvo_block_size;
    size_t one_meta_size = sizeof(DT) * (qProjSize * num_heads * qSize / 32);
    size_t q_offset_idx = meta_offset + group_idx * sizeof(DT);
    size_t q_scaling_idx = q_offset_idx + one_meta_size;

    size_t k_offset_idx = q_scaling_idx + one_meta_size;
    size_t k_scaling_idx = k_offset_idx + one_meta_size;

    size_t v_offset_idx = k_scaling_idx + one_meta_size;
    size_t v_scaling_idx = v_offset_idx + one_meta_size;

    size_t o_offset_idx = v_scaling_idx + one_meta_size;
    size_t o_scaling_idx = o_offset_idx + one_meta_size;

    weight_ptr[real_idx_q_first] =
        static_cast<DT>((input_weight_ptr[idx_q] >> 4) & 0xF) /
            (*(DT *)(input_weight_ptr + q_scaling_idx)) +
        (*(DT *)(input_weight_ptr + q_offset_idx));
    weight_ptr[real_idx_q_second] =
        static_cast<DT>((input_weight_ptr[idx_q] & 0xF)) /
            (*(DT *)(input_weight_ptr + q_scaling_idx + sizeof(DT))) +
        (*(DT *)(input_weight_ptr + q_offset_idx + sizeof(DT)));
    weight_ptr[real_idx_k_first] =
        static_cast<DT>((input_weight_ptr[idx_k] >> 4) & 0xF) /
            (*(DT *)(input_weight_ptr + k_scaling_idx)) +
        (*(DT *)(input_weight_ptr + k_offset_idx));
    weight_ptr[real_idx_k_second] =
        static_cast<DT>((input_weight_ptr[idx_k] & 0xF)) /
            (*(DT *)(input_weight_ptr + k_scaling_idx + sizeof(DT))) +
        (*(DT *)(input_weight_ptr + k_offset_idx + sizeof(DT)));
    weight_ptr[real_idx_v_first] =
        static_cast<DT>((input_weight_ptr[idx_v] >> 4) & 0xF) /
            (*(DT *)(input_weight_ptr + v_scaling_idx)) +
        (*(DT *)(input_weight_ptr + v_offset_idx));
    weight_ptr[real_idx_v_second] =
        static_cast<DT>((input_weight_ptr[idx_v] & 0xF)) /
            (*(DT *)(input_weight_ptr + v_scaling_idx + sizeof(DT))) +
        (*(DT *)(input_weight_ptr + v_offset_idx + sizeof(DT)));
    weight_ptr[real_idx_o_first] =
        static_cast<DT>((input_weight_ptr[idx_o] >> 4) & 0xF) /
            (*(DT *)(input_weight_ptr + o_scaling_idx)) +
        (*(DT *)(input_weight_ptr + o_offset_idx));
    weight_ptr[real_idx_o_second] =
        static_cast<DT>((input_weight_ptr[idx_o] & 0xF)) /
            (*(DT *)(input_weight_ptr + o_scaling_idx + sizeof(DT))) +
        (*(DT *)(input_weight_ptr + o_offset_idx + sizeof(DT)));
  }
}

template <typename DT>
__global__ void decompress_int8_attention_weights(char *input_weight_ptr,
                                                  DT *weight_ptr,
                                                  int qProjSize,
                                                  int qSize,
                                                  int num_heads) {
  // TODO this is because in top level function we assume q,k,v in same size
  CUDA_KERNEL_LOOP(i, qProjSize * num_heads * qSize) {
    int q_block_size = qProjSize * qSize;
    size_t qkvo_block_size = q_block_size * 4;

    int group_idx =
        (i / (INT4_NUM_OF_ELEMENTS_PER_GROUP * qSize)) * qSize + i % qSize;
    // i * 2 / (INT4_NUM_OF_ELEMENTS_PER_GROUP);
    int head_idx = i / q_block_size;
    int data_idx = i % q_block_size;

    size_t idx_q = head_idx * qkvo_block_size + data_idx;
    size_t idx_k = idx_q + q_block_size;
    size_t idx_v = idx_k + q_block_size;
    size_t idx_o = idx_v + q_block_size;

    size_t meta_offset = num_heads * qkvo_block_size;
    size_t one_meta_size = sizeof(DT) * (qProjSize * num_heads * qSize / 32);
    size_t q_offset_idx = meta_offset + group_idx * sizeof(DT);
    size_t q_scaling_idx = q_offset_idx + one_meta_size;

    size_t k_offset_idx = q_scaling_idx + one_meta_size;
    size_t k_scaling_idx = k_offset_idx + one_meta_size;

    size_t v_offset_idx = k_scaling_idx + one_meta_size;
    size_t v_scaling_idx = v_offset_idx + one_meta_size;

    size_t o_offset_idx = v_scaling_idx + one_meta_size;
    size_t o_scaling_idx = o_offset_idx + one_meta_size;

    weight_ptr[idx_q] = static_cast<DT>(input_weight_ptr[idx_q] & 0xFF) /
                            (*(DT *)(input_weight_ptr + q_scaling_idx)) +
                        (*(DT *)(input_weight_ptr + q_offset_idx));
    weight_ptr[idx_k] = static_cast<DT>(input_weight_ptr[idx_k] & 0xFF) /
                            (*(DT *)(input_weight_ptr + k_scaling_idx)) +
                        (*(DT *)(input_weight_ptr + k_offset_idx));
    weight_ptr[idx_v] = static_cast<DT>(input_weight_ptr[idx_v] & 0xFF) /
                            (*(DT *)(input_weight_ptr + v_scaling_idx)) +
                        (*(DT *)(input_weight_ptr + v_offset_idx));
    weight_ptr[idx_o] = static_cast<DT>(input_weight_ptr[idx_o] & 0xFF) /
                            (*(DT *)(input_weight_ptr + o_scaling_idx)) +
                        (*(DT *)(input_weight_ptr + o_offset_idx));
  }
}

template __global__ void decompress_int4_general_weights<float>(
    char const *input_weight_ptr, float *weight_ptr, int in_dim, int valueSize);
template __global__ void decompress_int4_general_weights<half>(
    char const *input_weight_ptr, half *weight_ptr, int in_dim, int valueSize);
template __global__ void decompress_int8_general_weights<float>(
    char const *input_weight_ptr, float *weight_ptr, int in_dim, int valueSize);
template __global__ void decompress_int8_general_weights<half>(
    char const *input_weight_ptr, half *weight_ptr, int in_dim, int valueSize);
template __global__ void
    decompress_int4_attention_weights<float>(char *input_weight_ptr,
                                             float *weight_ptr,
                                             int qProjSize,
                                             int qSize,
                                             int num_heads);

template __global__ void
    decompress_int4_attention_weights<half>(char *input_weight_ptr,
                                            half *weight_ptr,
                                            int qProjSize,
                                            int qSize,
                                            int num_heads);

template __global__ void
    decompress_int8_attention_weights<float>(char *input_weight_ptr,
                                             float *weight_ptr,
                                             int qProjSize,
                                             int qSize,
                                             int num_heads);

template __global__ void
    decompress_int8_attention_weights<half>(char *input_weight_ptr,
                                            half *weight_ptr,
                                            int qProjSize,
                                            int qSize,
                                            int num_heads);
// template <typename T1, typename T2>
// void decompress_weight_bias(T1 *input_weight_ptr,
//                             T2 *weight_ptr,
//                             T2 *params,
//                             int group_size,
//                             int tensor_size) {

//   // convert to DT, scaling, add offset;
//   cudaStream_t stream;
//   checkCUDA(get_legion_stream(&stream));
//   int parallelism = tensor_size;
//   decompress_kernel<<<GET_BLOCKS(parallelism),
//                       min(CUDA_NUM_THREADS, parallelism),
//                       0,
//                       stream>>>(
//       input_weight_ptr, weight_ptr, params, group_size);
// }
} // namespace Kernels
}; // namespace FlexFlow

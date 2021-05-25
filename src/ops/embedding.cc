/* Copyright 2019 Stanford
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
#include <assert.h>
#include <iostream>
#ifdef FF_USE_AVX2
#include <immintrin.h>
#endif

void EmbeddingLookup_int64_t_float_float__avx2_fma(
    const int block_size,
    const int output_size,
    const int index_size,
    const int data_size,
    const float* input,
    const int64_t* indices,
    const int* lengths,
    const float* weight,
    bool normalize_by_lengths,
    float* out) 
{
#ifdef FF_USE_AVX2
  const int64_t prefdist_T0 = 16;
  if (block_size == 128) {
    // unrolling 16 times
    int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      __m256 vop0 = _mm256_setzero_ps();
      __m256 vop8 = _mm256_setzero_ps();
      __m256 vop16 = _mm256_setzero_ps();
      __m256 vop24 = _mm256_setzero_ps();
      __m256 vop32 = _mm256_setzero_ps();
      __m256 vop40 = _mm256_setzero_ps();
      __m256 vop48 = _mm256_setzero_ps();
      __m256 vop56 = _mm256_setzero_ps();
      __m256 vop64 = _mm256_setzero_ps();
      __m256 vop72 = _mm256_setzero_ps();
      __m256 vop80 = _mm256_setzero_ps();
      __m256 vop88 = _mm256_setzero_ps();
      __m256 vop96 = _mm256_setzero_ps();
      __m256 vop104 = _mm256_setzero_ps();
      __m256 vop112 = _mm256_setzero_ps();
      __m256 vop120 = _mm256_setzero_ps();
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
        _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
        vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
        _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
        vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
        _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
        vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
        _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
        vop32 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (32)), vop32);
        _mm_prefetch((&ip_next_T0[32]), _MM_HINT_T0);
        vop40 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (40)), vop40);
        _mm_prefetch((&ip_next_T0[40]), _MM_HINT_T0);
        vop48 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (48)), vop48);
        _mm_prefetch((&ip_next_T0[48]), _MM_HINT_T0);
        vop56 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (56)), vop56);
        _mm_prefetch((&ip_next_T0[56]), _MM_HINT_T0);
        vop64 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (64)), vop64);
        _mm_prefetch((&ip_next_T0[64]), _MM_HINT_T0);
        vop72 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (72)), vop72);
        _mm_prefetch((&ip_next_T0[72]), _MM_HINT_T0);
        vop80 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (80)), vop80);
        _mm_prefetch((&ip_next_T0[80]), _MM_HINT_T0);
        vop88 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (88)), vop88);
        _mm_prefetch((&ip_next_T0[88]), _MM_HINT_T0);
        vop96 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (96)), vop96);
        _mm_prefetch((&ip_next_T0[96]), _MM_HINT_T0);
        vop104 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (104)), vop104);
        _mm_prefetch((&ip_next_T0[104]), _MM_HINT_T0);
        vop112 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (112)), vop112);
        _mm_prefetch((&ip_next_T0[112]), _MM_HINT_T0);
        vop120 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (120)), vop120);
        _mm_prefetch((&ip_next_T0[120]), _MM_HINT_T0);
      }
      if (normalize_by_lengths == false) {
        _mm256_storeu_ps(&op[0], vop0);
        _mm256_storeu_ps(&op[8], vop8);
        _mm256_storeu_ps(&op[16], vop16);
        _mm256_storeu_ps(&op[24], vop24);
        _mm256_storeu_ps(&op[32], vop32);
        _mm256_storeu_ps(&op[40], vop40);
        _mm256_storeu_ps(&op[48], vop48);
        _mm256_storeu_ps(&op[56], vop56);
        _mm256_storeu_ps(&op[64], vop64);
        _mm256_storeu_ps(&op[72], vop72);
        _mm256_storeu_ps(&op[80], vop80);
        _mm256_storeu_ps(&op[88], vop88);
        _mm256_storeu_ps(&op[96], vop96);
        _mm256_storeu_ps(&op[104], vop104);
        _mm256_storeu_ps(&op[112], vop112);
        _mm256_storeu_ps(&op[120], vop120);
      } else if (lengths[rangeIndex]) {
        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
        _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
        _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
        _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
        _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
        _mm256_storeu_ps(&op[32], _mm256_mul_ps(vop32, vlen_inv));
        _mm256_storeu_ps(&op[40], _mm256_mul_ps(vop40, vlen_inv));
        _mm256_storeu_ps(&op[48], _mm256_mul_ps(vop48, vlen_inv));
        _mm256_storeu_ps(&op[56], _mm256_mul_ps(vop56, vlen_inv));
        _mm256_storeu_ps(&op[64], _mm256_mul_ps(vop64, vlen_inv));
        _mm256_storeu_ps(&op[72], _mm256_mul_ps(vop72, vlen_inv));
        _mm256_storeu_ps(&op[80], _mm256_mul_ps(vop80, vlen_inv));
        _mm256_storeu_ps(&op[88], _mm256_mul_ps(vop88, vlen_inv));
        _mm256_storeu_ps(&op[96], _mm256_mul_ps(vop96, vlen_inv));
        _mm256_storeu_ps(&op[104], _mm256_mul_ps(vop104, vlen_inv));
        _mm256_storeu_ps(&op[112], _mm256_mul_ps(vop112, vlen_inv));
        _mm256_storeu_ps(&op[120], _mm256_mul_ps(vop120, vlen_inv));
      }
    }
  } else if (block_size == 64) {
    // unrolling 8 times
	  int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      __m256 vop0 = _mm256_setzero_ps();
      __m256 vop8 = _mm256_setzero_ps();
      __m256 vop16 = _mm256_setzero_ps();
      __m256 vop24 = _mm256_setzero_ps();
      __m256 vop32 = _mm256_setzero_ps();
      __m256 vop40 = _mm256_setzero_ps();
      __m256 vop48 = _mm256_setzero_ps();
      __m256 vop56 = _mm256_setzero_ps();
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
        _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
        vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
        _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
        vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
        _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
        vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
        _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
        vop32 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (32)), vop32);
        _mm_prefetch((&ip_next_T0[32]), _MM_HINT_T0);
        vop40 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (40)), vop40);
        _mm_prefetch((&ip_next_T0[40]), _MM_HINT_T0);
        vop48 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (48)), vop48);
        _mm_prefetch((&ip_next_T0[48]), _MM_HINT_T0);
        vop56 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (56)), vop56);
        _mm_prefetch((&ip_next_T0[56]), _MM_HINT_T0);
      }
      if (normalize_by_lengths == false) {
        _mm256_storeu_ps(&op[0], vop0);
        _mm256_storeu_ps(&op[8], vop8);
        _mm256_storeu_ps(&op[16], vop16);
        _mm256_storeu_ps(&op[24], vop24);
        _mm256_storeu_ps(&op[32], vop32);
        _mm256_storeu_ps(&op[40], vop40);
        _mm256_storeu_ps(&op[48], vop48);
        _mm256_storeu_ps(&op[56], vop56);
      } else if (lengths[rangeIndex]) {
        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
        _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
        _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
        _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
        _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
        _mm256_storeu_ps(&op[32], _mm256_mul_ps(vop32, vlen_inv));
        _mm256_storeu_ps(&op[40], _mm256_mul_ps(vop40, vlen_inv));
        _mm256_storeu_ps(&op[48], _mm256_mul_ps(vop48, vlen_inv));
        _mm256_storeu_ps(&op[56], _mm256_mul_ps(vop56, vlen_inv));
      }
    }
  } else if (block_size == 32) {
    // unrolling 4 times
    int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      __m256 vop0 = _mm256_setzero_ps();
      __m256 vop8 = _mm256_setzero_ps();
      __m256 vop16 = _mm256_setzero_ps();
      __m256 vop24 = _mm256_setzero_ps();
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
        _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
        vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
        _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
        vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
        _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
        vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
        _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
      }
      if (normalize_by_lengths == false) {
        _mm256_storeu_ps(&op[0], vop0);
        _mm256_storeu_ps(&op[8], vop8);
        _mm256_storeu_ps(&op[16], vop16);
        _mm256_storeu_ps(&op[24], vop24);
      } else if (lengths[rangeIndex]) {
        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
        _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
        _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
        _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
        _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
      }
    }
  } else {
    // generic code
    int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      int j = 0;
      for (; j + 8 <= block_size; j += 8) {
        _mm256_storeu_ps(op + j, _mm256_setzero_ps());
      }
      for (; j < block_size; j++) {
        op[j] = 0.0f;
      }
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        j = 0;
        for (; j + 8 <= block_size; j += 8) {
          _mm256_storeu_ps(
              &op[j],
              _mm256_fmadd_ps(
                  vwgt, _mm256_loadu_ps(&ip[j]), _mm256_loadu_ps(&op[j])));
          _mm_prefetch((&ip_next_T0[j]), _MM_HINT_T0);
        }
        for (; j < block_size; j++) {
          op[j] += wgt * ip[j];
        }
      }
      if (normalize_by_lengths && lengths[rangeIndex]) {
        float len_inv = 1.0f / lengths[rangeIndex];
        __m256 vlen_inv = _mm256_set1_ps(len_inv);
        j = 0;
        for (; j + 8 <= block_size; j += 8) {
          _mm256_storeu_ps(
              &op[j], _mm256_mul_ps(_mm256_loadu_ps(&op[j]), vlen_inv));
        }
        for (; j < block_size; j++) {
          op[j] = len_inv * op[j];
        }
      }
    }
  }
#else
  assert(0);
#endif
}

void embed_forward(const int64_t* input,
		const int* lengths,
		float* output,
		const float* embed,
		int block_size,
		int output_size,
		int index_size,
		int data_size)
{
  EmbeddingLookup_int64_t_float_float__avx2_fma(
      block_size,
      output_size,
      index_size,
      data_size,
      embed,
      input,
      lengths,
      nullptr,
      false,
      output
      );
}

void embed_backward_generic(const int64_t* input,
                            const int* lengths,
                            const float* output,
                            float* embed,
                            int block_size,
                            int output_size,
                            int index_size,
                            int data_size)
{
  //FIXME: Not functionaly correct.
  for (int i=0; i<output_size * block_size; i++)
  {
    int idx = i / block_size;
    int off = i % block_size;
    int64_t wordIdx = input[idx];
    // FIXME: Need to be atomic depending on the strategy
    embed[wordIdx * block_size + off] +=  output[i];;
  }
}

void embed_backward(const int64_t* input,
		                const int* lengths,
                    const float* output,
                    float* embed,
                    int block_size,
                    int output_size,
                    int index_size,
                    int data_size)
{
  embed_backward_generic(input, lengths, output, embed, block_size, output_size, index_size, data_size);
}


void Embedding::forward_task_cpu(const Task *task,
                                 const std::vector<PhysicalRegion>& regions,
                                 Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const AccessorRO<int64_t, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_weight(regions[2], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_weight = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  coord_t batch_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  // Input and output have same batch size
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;
  // Weight and output have same out dim
  assert(out_dim == rect_weight.hi[1] - rect_weight.lo[1] + 1);
  //const int64_t* input = acc_input.ptr(rect_input);
  //float* output = acc_output.ptr(rect_output);
  //const float* weight = acc_weight.ptr(rect_weight);
  int block_size = out_dim;
  int output_size = batch_size;
  int data_size = 1000000; //FIXME
  // For now we are assuming the length is always 1
  int index_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t in_dim = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(in_dim == 1);
  std::vector<int> lengths(output_size, 1);
  embed_forward(
      acc_input.ptr(rect_input), lengths.data(), acc_output.ptr(rect_output),
      acc_weight.ptr(rect_weight),
      block_size, output_size, index_size, data_size
  );
}

void Embedding::backward_task_cpu(const Task *task,
                                  const std::vector<PhysicalRegion>& regions,
                                  Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const AccessorRO<int64_t, 2> acc_input(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRW<float, 2> acc_weight(regions[2], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_weight = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  coord_t batch_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  // Input and output have same batch size
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  //coord_t in_dim = rect_input.hi[0] - rect_input.lo[0] + 1;
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;
  // Weight and output have same out dim
  assert(out_dim == rect_weight.hi[1] - rect_weight.lo[1] + 1);
  //const int64_t* input = acc_input.ptr(rect_input);
  //const float* output = acc_output.ptr(rect_output);
  //float* weight = acc_weight.ptr(rect_weight);
  int block_size = out_dim;
  int output_size = batch_size;
  int index_size = rect_input.hi[1] - rect_input.lo[0] + 1;
  int data_size = 1000000; //FIXME
  std::vector<int> lengths(output_size, 1);
  embed_backward(
      acc_input.ptr(rect_input), lengths.data(), acc_output.ptr(rect_output),
      acc_weight.ptr(rect_weight),
      block_size, output_size, index_size, data_size
  );
}

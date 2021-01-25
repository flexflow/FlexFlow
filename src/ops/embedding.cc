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
		float* out){}

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
  const int64_t* input = acc_input.ptr(rect_input);
  float* output = acc_output.ptr(rect_output);
  const float* weight = acc_weight.ptr(rect_weight);
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
  coord_t in_dim = rect_input.hi[0] - rect_input.lo[0] + 1;
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;
  // Weight and output have same out dim
  assert(out_dim == rect_weight.hi[1] - rect_weight.lo[1] + 1);
  const int64_t* input = acc_input.ptr(rect_input);
  const float* output = acc_output.ptr(rect_output);
  float* weight = acc_weight.ptr(rect_weight);
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


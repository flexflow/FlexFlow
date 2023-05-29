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

#include "flexflow/ops/beam_topk.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

enum class HeapType { kMinHeap, kMaxHeap };
enum class PreferIndices { kLower, kHigher };

template <typename T>
struct Entry {
  int index;
  T value;
};

template <typename T>
struct LinearData {
  typedef Entry<T> Entry;

  __device__ Entry &operator[](std::size_t index) const {
    return data[index];
  }

  __device__ int get_index(int i) const {
    return data[i].index;
  }
  __device__ T get_value(int i) const {
    return data[i].value;
  }

  Entry *const data;
};

template <typename T>
struct IndirectLinearData {
  typedef Entry<T> Entry;

  __device__ Entry &operator[](std::size_t index) const {
    return data[index];
  }

  __device__ int get_index(int i) const {
    return backing_data[data[i].index].index;
  }
  __device__ T get_value(int i) const {
    return data[i].value;
  }

  Entry *const data;
  Entry *const backing_data;
};

template <typename T>
struct StridedData {
  typedef Entry<T> Entry;

  __device__ Entry &operator[](std::size_t index) const {
    return data[index * blockDim.x + threadIdx.x];
  }

  __device__ int get_index(int i) const {
    return (*this)[i].index;
  }
  __device__ T get_value(int i) const {
    return (*this)[i].value;
  }

  Entry *const data;
};

// A heap of Entry<T> that can either work as a min-heap or as a max-heap.
template <HeapType heapType,
          PreferIndices preferIndices,
          template <typename>
          class Data,
          typename T>
struct IndexedHeap {
  typedef typename Data<T>::Entry Entry;
  Data<T> const data;
  __device__ IndexedHeap(Data<T> const &d) : data(d) {}

  __device__ bool is_above(int left, int right) {
    T left_value = data.get_value(left);
    T right_value = data.get_value(right);
    if (left_value == right_value) {
      if (preferIndices == PreferIndices::kLower) {
        return data.get_index(left) < data.get_index(right);
      } else {
        return data.get_index(left) > data.get_index(right);
      }
    }
    if (heapType == HeapType::kMinHeap) {
      return left_value < right_value;
    } else {
      return left_value > right_value;
    }
  }

  __device__ void assign(int i, Entry const &entry) {
    data[i] = entry;
  }

  __device__ void push_up(int i) {
    int child = i;
    int parent;
    for (; child > 0; child = parent) {
      parent = (child - 1) / 2;
      if (!is_above(child, parent)) {
        // Heap property satisfied.
        break;
      }
      swap(child, parent);
    }
  }

  __device__ void swap(int a, int b) {
    auto tmp = data[b];
    data[b] = data[a];
    data[a] = tmp;
  }

  __device__ void push_root_down(int k) {
    push_down(0, k);
  }

  // MAX-HEAPIFY in Cormen
  __device__ void push_down(int node, int k) {
    while (true) {
      int const left = 2 * node + 1;
      int const right = left + 1;
      int smallest = node;
      if (left < k && is_above(left, smallest)) {
        smallest = left;
      }
      if (right < k && is_above(right, smallest)) {
        smallest = right;
      }
      if (smallest == node) {
        break;
      }
      swap(smallest, node);
      node = smallest;
    }
  }

  // BUILD-MAX-HEAPIFY in Cormen
  __device__ void build(int k) {
    for (int node = (k - 1) / 2; node >= 0; node--) {
      push_down(node, k);
    }
  }

  // HEAP-EXTRACT-MAX in Cormen
  __device__ void remove_root(int k) {
    data[0] = data[k - 1];
    push_root_down(k - 1);
  }

  // in-place HEAPSORT in Cormen
  // This method destroys the heap property.
  __device__ void sort(int k) {
    for (int slot = k - 1; slot > 0; slot--) {
      // This is like remove_root but we insert the element at the end.
      swap(slot, 0);
      // Heap is now an element smaller.
      push_root_down(/*k=*/slot);
    }
  }

  __device__ void replace_root(Entry const &entry, int k) {
    data[0] = entry;
    push_root_down(k);
  }

  __device__ Entry const &root() {
    return data[0];
  }
};

template <HeapType heapType,
          PreferIndices preferIndices,
          template <typename>
          class Data,
          typename T>
__device__ IndexedHeap<heapType, preferIndices, Data, T>
    make_indexed_heap(typename Data<T>::Entry *data) {
  return IndexedHeap<heapType, preferIndices, Data, T>{Data<T>{data}};
}

// heapBeamTopK walks over [input, input+length) with `step_size` stride
// starting at `start_index`. It builds a top-`k` heap that is stored in
// `heap_entries` using `Accessor` to access elements in `heap_entries`. If
// sorted=true, the elements will be sorted at the end.
template <typename T, template <typename> class Data = LinearData>
__device__ void heapBeamTopK(T const *__restrict__ input,
                             int batch_index,
                             int length,
                             int k,
                             Entry<T> *__restrict__ heap_entries,
                             bool sorted = false,
                             int start_index = 0,
                             int step_size = 1) {
  assert(k <= length);
  auto heap =
      make_indexed_heap<HeapType::kMinHeap, PreferIndices::kHigher, Data, T>(
          heap_entries);

  int heap_end_index = start_index + k * step_size;
  if (heap_end_index > length) {
    heap_end_index = length;
  }
  // Initialize the min-heap.
  for (int index = start_index, slot = 0; index < heap_end_index;
       index += step_size, slot++) {
    heap.assign(slot, {index, input[index]});
  }

  heap.build(k);

  // Now iterate over the remaining items.
  // If an item is smaller than the min element, it is not amongst the top k.
  // Otherwise, replace the min element with it and push upwards.
  for (int index = heap_end_index; index < length; index += step_size) {
    // We prefer elements with lower indices. This is given here.
    // Later elements automatically have higher indices, so can be discarded.
    if (input[index] > heap.root().value) {
      // This element should replace the min.
      heap.replace_root({index, input[index]}, k);
    }
  }

  // Sort if wanted.
  if (sorted) {
    heap.sort(k);
  }

  // if(batch_index == 0){
  //   printf("top elemmments: %d, value %.15f\n", start_index,
  //   heap.root().value);
  // }
}

template <typename T>
__device__ void mergeBeamShards(int num_shards,
                                int batch_index,
                                int k,
                                int max_heap_size,
                                int request_id,
                                int *parent_id,
                                float *probs,
                                Entry<T> *__restrict__ entries,
                                Entry<T> *__restrict__ top_k_heap,
                                T *top_k_values,
                                int *top_k_indices,
                                int *top_k_parents) {
  // If k < num_shards, we can use a min-heap with k elements to get the top k
  // of the sorted blocks.
  // If k > num_shards, we can initialize a min-heap with the top element from
  // each sorted block.
  int const heap_size = k < num_shards ? k : num_shards;
  // printf("see value: %f", entries[0].value);
  // Min-heap part.

  {
    auto min_heap = IndexedHeap<HeapType::kMinHeap,
                                PreferIndices::kHigher,
                                IndirectLinearData,
                                T>{IndirectLinearData<T>{top_k_heap, entries}};
    // Initialize the heap as a min-heap.
    for (int slot = 0; slot < heap_size; slot++) {
      // int beam = (slot % max_heap_size) / k;
      float prob = probs[request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH +
                         ((slot % max_heap_size) / k)];
      min_heap.assign(slot, {slot, (entries[slot].value * prob)});
      if (batch_index == 0) {
        printf("slot %d, value %.15f, prob %15f\n",
               slot,
               entries[slot].value,
               prob);
      }
    }
    min_heap.build(heap_size);

    // Now perform top k with the remaining shards (if num_shards > heap_size).
    for (int shard = heap_size; shard < num_shards; shard++) {
      auto const entry = entries[shard];
      auto const root = min_heap.root();

      float prob = probs[request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH +
                         ((shard % max_heap_size) / k)];
      if (batch_index == 0) {
        printf("shard %d, index %d, value %.15f, prob %.15f\n",
               shard,
               entry.index,
               entry.value,
               prob);
      }
      if (entry.value * prob < root.value) {
        continue;
      }
      if (entry.value * prob == root.value &&
          entry.index > entries[root.index].index) {
        continue;
      }
      // This element should replace the min.
      min_heap.replace_root({shard, entry.value * prob}, heap_size);
    }
  }

  // Max-part.
  {
    // Turn the min-heap into a max-heap in-place.
    auto max_heap = IndexedHeap<HeapType::kMaxHeap,
                                PreferIndices::kLower,
                                IndirectLinearData,
                                T>{IndirectLinearData<T>{top_k_heap, entries}};
    // Heapify into a max heap.
    max_heap.build(heap_size);

    // Now extract the minimum k-1 times.
    // k is treated specially.
    int const last_k = k - 1;
    for (int rank = 0; rank < last_k; rank++) {
      Entry<T> const &max_element = max_heap.root();
      top_k_values[rank] = max_element.value;
      int shard_index = max_element.index;
      top_k_indices[rank] = entries[shard_index].index;
      top_k_parents[rank] =
          parent_id[request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH +
                    ((shard_index % max_heap_size) / k)];
      int next_shard_index = shard_index + num_shards;

      float prob = probs[request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH +
                         ((next_shard_index % max_heap_size) / k)];
      if (batch_index == 0) {
        printf("next_shard_index %d, value %.15f, prob %.15f\n",
               next_shard_index,
               entries[next_shard_index].value,
               prob);
      }

      max_heap.replace_root(
          {next_shard_index, entries[next_shard_index].value * prob},
          heap_size);
    }

    // rank == last_k.
    Entry<T> const &max_element = max_heap.root();
    top_k_values[last_k] = max_element.value;
    int shard_index = max_element.index;
    top_k_indices[last_k] = entries[shard_index].index;
    top_k_parents[last_k] =
        parent_id[request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH +
                  ((shard_index % max_heap_size) / k)];
  }
}

template <typename T>
__global__ void
    mergeSubRequestsKernel(int64_t N, T const *X, T const *rstd, T *Y) {
  using T_ACC = T;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    Y[index] = static_cast<T_ACC>(X[index]) * static_cast<T_ACC>(rstd[i]);
  }
}

template <typename T>
__global__ void beam_topk_forward_kernel(T const *__restrict__ input,
                                         size_t shared_memory_size,
                                         int length,
                                         int k,
                                         int max_heap_size,
                                         int *parent_ids,
                                         float *acc_probs,
                                         int *gpu_block_start_index,
                                         int *gpu_request_id,
                                         int *tokens_per_request,
                                         bool sorted,
                                         T *__restrict__ output,
                                         int *__restrict__ indices,
                                         int *__restrict__ parents,
                                         bool is_print) {
  __shared__ char shared_memory[48 << 10];
  int const batch_index = blockIdx.x;
  // T const *batch_input = input + batch_index * length;
  int const thread_index = threadIdx.x;
  int const thread_count = blockDim.x;
  int const request_id = gpu_request_id[batch_index];
  int const token_nums = tokens_per_request[batch_index];
  Entry<T> *shared_entries = (Entry<T> *)shared_memory;

  int sub_request_id = thread_index / k;
  // if (is_print) {
  //   printf("beam kernel: batch_index: %d, thread_index %d, sub_request_id %d,
  //   "
  //          "request_id %d, token_nums %d\n",
  //          batch_index,
  //          thread_index,
  //          sub_request_id,
  //          request_id,
  //          token_nums);
  // }

  T const *batch_input = input + gpu_block_start_index[batch_index] +
                         (sub_request_id * token_nums * length);

  if (batch_index == 0) {
    printf("request 0 start index: thread index %d, offset %d, batch_input %p, "
           "acc index %d acc "
           "prob %f, thread_count %d, request_id %d\n",
           thread_index,
           gpu_block_start_index[batch_index] +
               (sub_request_id * token_nums * length),
           batch_input,
           request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH + sub_request_id,
           acc_probs[request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH +
                     sub_request_id],
           thread_count,
           request_id);
  }
  // printf("thread index %d, thread_count %d, batch_index %d\n", thread_index,
  // thread_count, batch_index);
  heapBeamTopK<T, StridedData>(batch_input,
                               batch_index,
                               length,
                               k,
                               shared_entries,
                               true,
                               thread_index % k,
                               k);
  __syncthreads();
  // printf("beam thread index %d, thread_count %d, thread index %d, batch_index
  // "
  //        "%d, k %d, parent_id %d, acc_prob: %f, sub id: %d, request_id: %d,
  //        offset: %d, offset2 %d, sub_request_id %d\n", thread_index,
  //        thread_count,
  //        thread_index,
  //        batch_index,
  //        k,
  //        parent_ids[request_id * BatchConfig::MAX_NUM_BEAMS +
  //        sub_request_id], acc_probs[request_id * BatchConfig::MAX_NUM_BEAMS +
  //        sub_request_id], sub_request_id, request_id,
  //        gpu_block_start_index[batch_index],
  //        batch_index * length,
  //        sub_request_id);

  if (thread_index == 0) {
    // merge beam_width heaps and store the parent
    // find which req it belongs to, replace the offset
    printf("merge heaps, batch index: %d, sub_request_id %d, value %f\n",
           batch_index,
           sub_request_id,
           acc_probs[request_id * BeamSearchBatchConfig::MAX_BEAM_WIDTH +
                     sub_request_id]);
    int const offset = batch_index * k;
    auto batch_output = output + offset;
    auto batch_indices = indices + offset;
    auto batch_parents = parents + offset;
    Entry<T> *top_k_heap = shared_entries + thread_count * k;

    // if(batch_index == 0 && is_print){
    //   for(int i = 0; i < 18; i++){
    //       printf("see value: %.15f\n", shared_entries[i].value);
    //   }
    // }

    // get parent/acc based on the sub request and main request
    mergeBeamShards(thread_count,
                    batch_index,
                    k,
                    max_heap_size,
                    request_id,
                    parent_ids,
                    acc_probs,
                    shared_entries,
                    top_k_heap,
                    batch_output,
                    batch_indices,
                    batch_parents);
  }
}

/*static*/
void BeamTopK::forward_kernel(BeamTopKMeta const *m,
                              BeamSearchBatchConfig const *bc,
                              float const *input_ptr,
                              float *output_ptr,
                              int *indices_ptr,
                              int *parent_ptr,
                              size_t batch_size,
                              int length,
                              bool sorted,
                              hipStream_t stream) {
  // Adopted from TensorFlow's BeamTopK implementation
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/topk_op_gpu.h

  int num_shards = 0;
  int max_heap_size = 0;
  int max_beam_width = 0;
  int req_index = 0;

  // sub request
  int const *sub_requests = bc->sub_requests;

  // std::vector<BatchConfig::BeamSlot> beam_slots = bc->beam_slots;
  // assert(bc->beam_slots.size() > 0);

  int beam_num_blocks = 0;
  std::vector<int> beam_block_start_index;
  std::vector<int> request_id;
  std::vector<int> tokens_per_request;

  int block_start_index = 0;
  int depth =
      bc->beamRequestsInfo[bc->tokensInfo[0].request_index].current_depth;

  // a data structure for prob, parent_id,
  int max_total_requests =
      BeamSearchBatchConfig::MAX_BEAM_WIDTH * bc->num_active_requests();
  int parent_ids[max_total_requests];
  float acc_probs[max_total_requests];

  for (int i = 0; i < bc->MAX_NUM_REQUESTS; i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    assert(bc->beamRequestsInfo[i].beam_size > 0);

    // int num_new_tokens = bc->num_processing_tokens[i];
    int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;

    // get beam size;
    int beam_size = bc->beamRequestsInfo[i].beam_size;

    // initial request
    std::cout << "sub_requests: " << i << ", " << sub_requests[i] << "\n";
    assert(sub_requests[i] > 0);
    // process sub requests
    for (int j = 0; j < sub_requests[i]; j++) {
      parent_ids[req_index * BeamSearchBatchConfig::MAX_BEAM_WIDTH + j] = j;
      // beam_slots[i].parent_id[j];
      acc_probs[req_index * BeamSearchBatchConfig::MAX_BEAM_WIDTH + j] =
          bc->beamRequestsInfo[i].probs[j];
      std::cout << "probbbb req: " << i << ", sub req probability : "
                << bc->beamRequestsInfo[i].probs[j] << ", sub request id " << j
                << ", parent id " << bc->beamRequestsInfo[i].parent_id[j]
                << ", data inddd"
                << req_index * BeamSearchBatchConfig::MAX_BEAM_WIDTH + j
                << "\n";
    }

    // process tokens
    for (int k = 0; k < num_new_tokens; k++) {
      beam_block_start_index.push_back(block_start_index);
      request_id.push_back(i);
      tokens_per_request.push_back(num_new_tokens);
      block_start_index += length;
      beam_num_blocks++;
    }

    max_heap_size = std::max(max_heap_size, beam_size * sub_requests[i]);
    max_beam_width = std::max(max_beam_width, beam_size);
    req_index += 1;
    block_start_index += (sub_requests[i] - 1) * num_new_tokens * length;
  }
  std::cout << "what index: " << block_start_index
            << ", block num: " << beam_num_blocks << "\n";

  assert(batch_size >= beam_num_blocks);
  assert(bc->num_active_requests() == req_index);

  {
    constexpr auto shared_memory_size = 48 << 10;
    auto const heap_size = max_heap_size * sizeof(Entry<float>);
    // shared_memory_size = (num_shards + 1) * heap_size <=>
    num_shards = shared_memory_size / heap_size - 1;
    assert(num_shards > 0);
    if (num_shards > CUDA_NUM_THREADS) {
      num_shards = CUDA_NUM_THREADS;
    }
    std::cout << "maxheap size:  " << max_heap_size << "\n";
    std::cout << "maxbeam width:  " << max_beam_width
              << ", heap size: " << heap_size << "\n";
  }
  // We are limited by the amount of shared memory we have per block.
  size_t shared_memory_size =
      (num_shards + 1) * max_heap_size * sizeof(Entry<float>);

  assert(num_shards >= (size_t)max_heap_size);
  num_shards = max_heap_size;

  // parent_id, per token
  int *gpu_parents;
  // acc_porbs, per token
  float *gpu_probs;
  // each block's start index;
  // one block means the single token in different requests;
  int *gpu_block_start_index;
  int *gpu_request_id;
  int *gpu_tokens_per_request;

  checkCUDA(hipMalloc(&gpu_parents, sizeof(int) * max_total_requests));
  checkCUDA(hipMalloc(&gpu_probs, sizeof(float) * max_total_requests));
  checkCUDA(hipMalloc(&gpu_block_start_index, sizeof(int) * beam_num_blocks));
  checkCUDA(hipMalloc(&gpu_request_id, sizeof(int) * beam_num_blocks));
  checkCUDA(hipMalloc(&gpu_tokens_per_request, sizeof(int) * beam_num_blocks));
  checkCUDA(hipMemcpy(gpu_parents,
                      parent_ids,
                      sizeof(int) * max_total_requests,
                      hipMemcpyHostToDevice));
  checkCUDA(hipMemcpy(gpu_probs,
                      acc_probs,
                      sizeof(float) * max_total_requests,
                      hipMemcpyHostToDevice));
  checkCUDA(hipMemcpy(gpu_block_start_index,
                      beam_block_start_index.data(),
                      sizeof(int) * beam_num_blocks,
                      hipMemcpyHostToDevice));
  checkCUDA(hipMemcpy(gpu_request_id,
                      request_id.data(),
                      sizeof(int) * beam_num_blocks,
                      hipMemcpyHostToDevice));
  checkCUDA(hipMemcpy(gpu_tokens_per_request,
                      tokens_per_request.data(),
                      sizeof(int) * beam_num_blocks,
                      hipMemcpyHostToDevice));

  beam_topk_forward_kernel<<<beam_num_blocks, num_shards, 0, stream>>>(
      input_ptr,
      shared_memory_size,
      length,
      max_beam_width,
      max_heap_size,
      gpu_parents,
      gpu_probs,
      gpu_block_start_index,
      gpu_request_id,
      gpu_tokens_per_request,
      sorted,
      output_ptr,
      indices_ptr,
      parent_ptr,
      depth == 1);

  // merge sub
}

/*static*/
void BeamTopK::forward_kernel_wrapper(BeamTopKMeta const *m,
                                      BeamSearchBatchConfig const *bc,
                                      float const *input_ptr,
                                      float *output_ptr,
                                      int *indices_ptr,
                                      int *parent_ptr,
                                      size_t batch_size,
                                      int length,
                                      bool sorted) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }

  BeamTopK::forward_kernel(m,
                           bc,
                           input_ptr,
                           output_ptr,
                           indices_ptr,
                           parent_ptr,
                           batch_size,
                           length,
                           sorted,
                           stream);

  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
    printf("[BeamTopK] forward time = %.2lfms\n", elapsed);
  }
  // if(bc->beam_slots.at(0).current_depth == 1){
  //     print_beam_tensor<float>((float *)input_ptr, 50, 32000, 15, "beam topk
  //     input"); print_tensor<float>((float *)output_ptr, 50, "beam topk
  //     output");
  // }
}

BeamTopKMeta::BeamTopKMeta(FFHandler handler, Op const *op)
    : OpMeta(handler, op) {}

}; // namespace FlexFlow

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
#include "flexflow/utils/cuda_helper.h"

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
                             int *request_length,
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

  // get data from k sub requests
  // Initialize the min-heap.
  T const *input_data = input;
  int slot = 0;
  for (int i = 0; i < k; i++) {
    for (int index = start_index; index < heap_end_index; index += step_size) {
      heap.assign(slot, {index, input_data[index]});
      slot++;
    }
    input_data += request_length[batch_index]  * length;
  }

  // // Initialize the min-heap.
  // for (int index = start_index, slot = 0; index < heap_end_index;
  //      index += step_size, slot++) {
  //   heap.assign(slot, {index, input[index]});
  // }

  heap.build(k * k);

  for (int i = 0; i < k; i++) {
    input_data = input;
    for (int index = heap_end_index; index < length; index += step_size) {
      // We prefer elements with lower indices. This is given here.
      // Later elements automatically have higher indices, so can be discarded.
      if (input_data[index] > heap.root().value) {
        // This element should replace the min.
        heap.replace_root({index, input_data[index]}, k);
      }
    }
    input_data += request_length[batch_index]  * length;
  }

  // Sort if wanted.
  if (sorted) {
    heap.sort(k * k);
  }
}

// mergeShards performs a top-k merge on `num_shards` many sorted streams that
// are sorted and stored in `entries` in a strided way:
// |s_1 1st|s_2 1st|...s_{num_shards} 1st|s_1 2nd|s_2 2nd|...
// The overall top k elements are written to `top_k_values` and their indices
// to top_k_indices.
// `top_k_heap` is used as temporary storage for the merge heap.
template <typename T>
__device__ void mergeShards(int num_shards,
                            int k,
                            Entry<T> *__restrict__ entries,
                            Entry<T> *__restrict__ top_k_heap,
                            // T *top_k_values,
                            int *top_k_indices) {
  // If k < num_shards, we can use a min-heap with k elements to get the top k
  // of the sorted blocks.
  // If k > num_shards, we can initialize a min-heap with the top element from
  // each sorted block.
  int const heap_size = k < num_shards ? k : num_shards;

  // Min-heap part.
  {
    auto min_heap = IndexedHeap<HeapType::kMinHeap,
                                PreferIndices::kHigher,
                                IndirectLinearData,
                                T>{IndirectLinearData<T>{top_k_heap, entries}};
    // Initialize the heap as a min-heap.
    for (int slot = 0; slot < heap_size; slot++) {
      min_heap.assign(slot, {slot, entries[slot].value});
    }
    min_heap.build(heap_size);

    // Now perform top k with the remaining shards (if num_shards > heap_size).
    for (int shard = heap_size; shard < num_shards; shard++) {
      auto const entry = entries[shard];
      auto const root = min_heap.root();
      if (entry.value < root.value) {
        continue;
      }
      if (entry.value == root.value &&
          entry.index > entries[root.index].index) {
        continue;
      }
      // This element should replace the min.
      min_heap.replace_root({shard, entry.value}, heap_size);
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
      // top_k_values[rank] = max_element.value;
      int shard_index = max_element.index;
      top_k_indices[rank] = entries[shard_index].index;
      int next_shard_index = shard_index + num_shards;
      // For rank < k-1, each top k heap still contains at least 1 element,
      // so we can draw a replacement.
      max_heap.replace_root({next_shard_index, entries[next_shard_index].value},
                            heap_size);
    }

    // rank == last_k.
    Entry<T> const &max_element = max_heap.root();
    // top_k_values[last_k] = max_element.value;
    int shard_index = max_element.index;
    top_k_indices[last_k] = entries[shard_index].index;
  }
}

template <typename T>
__global__ void beam_topk_forward_kernel(T const *__restrict__ input,
                                         size_t shared_memory_size,
                                         int length,
                                         int *heap_size,
                                         int *request_nums,
                                         int *request_index,
                                         int *request_length,
                                         bool sorted,
                                         // T *__restrict__ output,
                                         int *__restrict__ indices) {
  __shared__ char shared_memory[48 << 10];

  // //1， 2， 3... 20
  int const batch_index = blockIdx.x;
  int const thread_index = threadIdx.x;
  int const thread_count = blockDim.x;
  Entry<T> *shared_entries = (Entry<T> *)shared_memory;

  // heap size of a specific req
  int k = heap_size[batch_index];

  int pre_offset = 0;
  for(int i = 0; i < batch_index; i++){
      if(request_nums[i] < request_nums[batch_index]){
        pre_offset += heap_size[i] * length;
      }
  } 
  pre_offset += request_index[batch_index] * length ;

  printf("beam thread index %d, thread_count %d, batch_index %d, k %d, which "
         "req %d, req index %d, cuda_request_length %d, pre_offset %d\n",
         thread_index,
         thread_count,
         batch_index,
         k,
         request_nums[batch_index],
         request_index[batch_index],
         request_length[batch_index],
         pre_offset);

  heapBeamTopK<T, StridedData>(input + pre_offset,
                               request_length,
                               batch_index,
                               length,
                               k,
                               shared_entries,
                               true,
                               thread_index,
                               thread_count);

  __syncthreads();
  if (thread_index == 0) {
    int const offset = batch_index * k;
    // auto batch_output = output + offset;
    auto batch_indices = indices + offset;
    Entry<T> *top_k_heap = shared_entries + thread_count * k;
    mergeShards(thread_count,
                k,
                shared_entries,
                top_k_heap,
                // batch_output,
                batch_indices);
  }
}

/*static*/
void BeamTopK::forward_kernel(BeamTopKMeta const *m,
                              BatchConfig const *bc,
                              float const *input_ptr,
                              // float *output_ptr,
                              int *indices_ptr,
                              size_t batch_size,
                              size_t tokens_per_request,
                              int length,
                              bool sorted,
                              cudaStream_t stream) {
  // Adopted from TensorFlow's BeamTopK implementation
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/topk_op_gpu.h
  int num_shards = 0;
  int max_heap_size = 0;
  int index = 0;

  // sub request
  size_t beam_num_blocks = 0;
  std::unordered_map<size_t, int> sub_requests = bc->sub_requests;

  std::vector<int> heap_sizes;
  std::vector<int> request_num;
  std::vector<int> request_index;
  std::vector<int> request_length;

  for (int i = 0; i < bc->MAX_NUM_REQUESTS; i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    int num_new_tokens = bc->num_processing_tokens[i];
    batch_size -= num_new_tokens * sub_requests.at(i);
    beam_num_blocks += num_new_tokens;

    for (int j = 0; j < num_new_tokens; j++) {
      heap_sizes.push_back(sub_requests.at(i));
      request_num.push_back(i);
      request_index.push_back(j);
      request_length.push_back(num_new_tokens);
    }

    assert(sub_requests.at(i) > -1);
    int beam_heap_length = sub_requests[i] * sub_requests[i];
    tokens[index] = bc->num_processing_tokens[i];
    max_heap_size = std::max(max_heap_size, beam_heap_length);

    index += 1;

    for (int i = 0; i < bc->num_processing_tokens[i]; i++) {
    }
  }
  assert(batch_size == 0);
  // std::cout << "heap sizesssss:  " << heap_sizes.size() << "\n";
  assert(index == bc->num_active_requests());

  int *cuda_heaps;

  //token in which requests
  int *cuda_requests;

  //token in request's index
  int *cuda_requests_index;

  //how many tokens in the token's request
  int *cuda_request_length;
  checkCUDA(cudaMalloc(&cuda_heaps, sizeof(int) * heap_sizes.size()));
  checkCUDA(cudaMalloc(&cuda_requests, sizeof(int) * heap_sizes.size()));
  checkCUDA(cudaMalloc(&cuda_requests_index, sizeof(int) * heap_sizes.size()));
  checkCUDA(cudaMalloc(&cuda_request_length, sizeof(int) * heap_sizes.size()));

  checkCUDA(cudaMemcpy(cuda_heaps,
                       heap_sizes.data(),
                       sizeof(int) * heap_sizes.size(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(cuda_requests,
                       request_num.data(),
                       sizeof(int) * heap_sizes.size(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(cuda_requests_index,
                       request_index.data(),
                       sizeof(int) * heap_sizes.size(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(cuda_request_length,
                       request_length.data(),
                       sizeof(int) * heap_sizes.size(),
                       cudaMemcpyHostToDevice));                     
                       

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
    std::cout << "maxheap size:  " << max_heap_size << "heap size, "
              << heap_size << "\n";
  }
  // We are limited by the amount of shared memory we have per block.
  size_t shared_memory_size =
      (num_shards + 1) * max_heap_size * sizeof(Entry<float>);


  assert(num_shards >= (size_t)max_heap_size);
  num_shards = max_heap_size;

  std::cout << "num blocks:  " << beam_num_blocks << "num shards,  "
            << num_shards << "\n";

  beam_topk_forward_kernel<<<beam_num_blocks, num_shards, 0, stream>>>(
      input_ptr,
      shared_memory_size,
      length,
      cuda_heaps,
      cuda_requests,
      cuda_requests_index,
      cuda_request_length,
      sorted,
      // output_ptr,
      indices_ptr);
}

/*static*/
void BeamTopK::forward_kernel_wrapper(BeamTopKMeta const *m,
                                      BatchConfig const *bc,
                                      float const *input_ptr,
                                      // float *output_ptr,
                                      int *indices_ptr,
                                      size_t batch_size,
                                      size_t tokens_per_request,
                                      int length,
                                      bool sorted) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  BeamTopK::forward_kernel(m,
                           bc,
                           input_ptr,
                           // output_ptr,
                           indices_ptr,
                           batch_size,
                           tokens_per_request,
                           length,
                           sorted,
                           stream);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[BeamTopK] forward time = %.2lfms\n", elapsed);
  }
}

BeamTopKMeta::BeamTopKMeta(FFHandler handler) : OpMeta(handler) {}

}; // namespace FlexFlow

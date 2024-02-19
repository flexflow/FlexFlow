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

#include "flexflow/ops/arg_topk.h"
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

// heapArgTopK walks over [input, input+length) with `step_size` stride starting
// at `start_index`. It builds a top-`k` heap that is stored in `heap_entries`
// using `Accessor` to access elements in `heap_entries`. If sorted=true, the
// elements will be sorted at the end.
template <typename T, template <typename> class Data = LinearData>
__device__ void heapArgTopK(T const *__restrict__ input,
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
                            float *top_k_values,
                            int *top_k_indices,
                            bool speculative_decoding) {
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
      if (speculative_decoding) {
        assert(top_k_values != nullptr);
        top_k_values[rank] = static_cast<float>(max_element.value);
      }
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
__global__ void arg_topk_forward_kernel(T const *__restrict__ input,
                                        size_t shared_memory_size,
                                        int length,
                                        int k,
                                        bool sorted,
                                        float *__restrict__ output,
                                        int *__restrict__ indices,
                                        bool speculative_decoding) {
  __shared__ char shared_memory[48 << 10];
  int const batch_index = blockIdx.x;
  T const *batch_input = input + batch_index * length;
  int const thread_index = threadIdx.x;
  int const thread_count = blockDim.x;
  Entry<T> *shared_entries = (Entry<T> *)shared_memory;
  heapArgTopK<T, StridedData>(
      batch_input, length, k, shared_entries, true, thread_index, thread_count);
  __syncthreads();
  if (thread_index == 0) {
    int const offset = batch_index * k;
    auto batch_output = output + offset;
    auto batch_indices = indices + offset;
    Entry<T> *top_k_heap = shared_entries + thread_count * k;
    mergeShards(thread_count,
                k,
                shared_entries,
                top_k_heap,
                batch_output,
                batch_indices,
                speculative_decoding);
  }
}

/*static*/
template <typename DT>
void ArgTopK::forward_kernel(ArgTopKMeta const *m,
                             DT const *input_ptr,
                             float *output_ptr,
                             int *indices_ptr,
                             size_t batch_size,
                             int length,
                             int k,
                             bool sorted,
                             BeamSearchBatchConfig const *bc,
                             hipStream_t stream) {
  // Adopted from TensorFlow's ArgTopK implementation
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/topk_op_gpu.h
  int num_shards = 0;
  {
    constexpr auto shared_memory_size = 48 << 10;
    auto const heap_size = k * sizeof(Entry<DT>);
    // shared_memory_size = (num_shards + 1) * heap_size <=>
    num_shards = shared_memory_size / heap_size - 1;
    assert(num_shards > 0);
    if (num_shards > CUDA_NUM_THREADS) {
      num_shards = CUDA_NUM_THREADS;
    }
  }
  // We are limited by the amount of shared memory we have per block.
  size_t shared_memory_size = (num_shards + 1) * k * sizeof(Entry<DT>);
  // size_t num_blocks = (batch_size + num_shards - 1) / num_shards;
  size_t num_blocks = batch_size;
  // all requests are in the same beam stages
  if (m->speculative_decoding) {
    assert(bc->num_active_requests() >= 0);

    // check
    int beam_size = -1;
    for (int i = 1; i < bc->max_requests_per_batch(); i++) {
      if (bc->request_completed[i]) {
        continue;
      } else if (beam_size == -1) {
        beam_size = bc->beamRequestsInfo[i].beam_size;
      } else {
        assert(beam_size == bc->beamRequestsInfo[i].beam_size);
      }
    }

    assert(num_shards >= (size_t)beam_size);
    num_shards = k;
    arg_topk_forward_kernel<<<num_blocks, num_shards, 0, stream>>>(
        input_ptr,
        shared_memory_size,
        length,
        beam_size,
        sorted,
        output_ptr,
        indices_ptr,
        m->speculative_decoding);
  } else {

    assert(num_shards >= (size_t)k);
    num_shards = k;
    arg_topk_forward_kernel<<<num_blocks, num_shards, 0, stream>>>(
        input_ptr,
        shared_memory_size,
        length,
        k,
        sorted,
        nullptr,
        indices_ptr,
        false);
  }
}

/*static*/
void ArgTopK::forward_kernel_wrapper(ArgTopKMeta const *m,
                                     GenericTensorAccessorR const &input,
                                     GenericTensorAccessorW const &probs,
                                     // float *output_ptr,
                                     GenericTensorAccessorW const &indices,
                                     int batch_size,
                                     BeamSearchBatchConfig const *bc) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Domain in1_domain = runtime->get_index_space_domain(
  //     ctx, task->regions[0].region.get_index_space());
  //   Domain out1_domain = runtime->get_index_space_domain(
  //       ctx, task->regions[1].region.get_index_space());
  // Domain out2_domain = runtime->get_index_space_domain(
  //     ctx, task->regions[1].region.get_index_space());
  int numdims = input.domain.get_dim();
  assert(indices.domain.get_dim() == numdims);

  int in_cols = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  // int out1_cols = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  int out2_cols = indices.domain.hi()[0] - indices.domain.lo()[0] + 1;

  // assert(out1_domain == out2_domain);
  for (int i = 1; i < input.domain.get_dim(); i++) {
    assert(input.domain.lo()[i] == indices.domain.lo()[i]);
    assert(input.domain.hi()[i] == indices.domain.hi()[i]);
  }
  // float const *in_ptr = helperGetTensorPointerRO<float>(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  //   float *value_ptr = helperGetTensorPointerWO<float>(
  //       regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // int *index_ptr = helperGetTensorPointerWO<int>(
  //    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  int length = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int k = indices.domain.hi()[0] - indices.domain.lo()[0] +
          1; /*TODO: This prints to 5*/
  // size_t batch_size = input.domain.get_volume() / length;
  // assert(indices.domain.get_volume() / k == batch_size);

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA((&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  if (input.data_type == DT_HALF) {
    ArgTopK::forward_kernel(m,
                            input.get_half_ptr(),
                            // output_ptr,
                            m->speculative_decoding ? probs.get_float_ptr()
                                                    : nullptr,
                            indices.get_int32_ptr(),
                            batch_size,
                            length,
                            k,
                            m->sorted,
                            m->speculative_decoding ? bc : nullptr,
                            stream);
  } else if (input.data_type == DT_FLOAT) {
    ArgTopK::forward_kernel(m,
                            input.get_float_ptr(),
                            // output_ptr,
                            m->speculative_decoding ? probs.get_float_ptr()
                                                    : nullptr,
                            indices.get_int32_ptr(),
                            batch_size,
                            length,
                            k,
                            m->sorted,
                            m->speculative_decoding ? bc : nullptr,
                            stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
  }
}

ArgTopKMeta::ArgTopKMeta(FFHandler handler, Op const *op)
    : OpMeta(handler, op) {}

}; // namespace FlexFlow

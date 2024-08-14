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

#include "flexflow/ops/gumbel_topk.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

enum class HeapType { kMinHeap, kMaxHeap };
enum class PreferIndices { kLower, kHigher };

template <typename T>
struct GumbelEntry {
  int index;
  T value;
  T perturbed_value;
};

template <typename T>
struct LinearData {
  typedef GumbelEntry<T> GumbelEntry;

  __device__ GumbelEntry &operator[](std::size_t i) const {
    return data[i];
  }

  __device__ int get_index(int i) const {
    return data[i].index;
  }
  __device__ T get_value(int i) const {
    return data[i].value;
  }
  __device__ T get_perturbed_value(int i) const {
    return data[i].perturbed_value;
  }

  GumbelEntry *const data;
};

template <typename T>
struct IndirectLinearData {
  typedef GumbelEntry<T> GumbelEntry;

  __device__ GumbelEntry &operator[](std::size_t i) const {
    return data[i];
  }

  __device__ int get_index(int i) const {
    return backing_data[data[i].index].index;
  }
  __device__ T get_value(int i) const {
    return data[i].value;
  }
  __device__ T get_perturbed_value(int i) const {
    return data[i].perturbed_value;
  }

  GumbelEntry *const data;
  GumbelEntry *const backing_data;
};

template <typename T>
struct StridedData {
  typedef GumbelEntry<T> GumbelEntry;

  __device__ GumbelEntry &operator[](std::size_t i) const {
    return data[i * blockDim.x + threadIdx.x];
  }

  __device__ int get_index(int i) const {
    return (*this)[i].index;
  }
  __device__ T get_value(int i) const {
    return (*this)[i].value;
  }
  __device__ T get_perturbed_value(int i) const {
    return (*this)[i].perturbed_value;
  }

  GumbelEntry *const data;
};

// A heap of GumbelEntry<T> that can either work as a min-heap or as a max-heap.
template <HeapType heapType,
          PreferIndices preferIndices,
          template <typename>
          class Data,
          typename T>
struct IndexedHeap {
  typedef typename Data<T>::GumbelEntry GumbelEntry;
  Data<T> const data;
  __device__ IndexedHeap(Data<T> const &d) : data(d) {}

  __device__ bool is_above(int left, int right) {
    T left_perturbed_value = data.get_perturbed_value(left);
    T right_perturbed_value = data.get_perturbed_value(right);
    if (left_perturbed_value == right_perturbed_value) {
      if (preferIndices == PreferIndices::kLower) {
        return data.get_index(left) < data.get_index(right);
      } else {
        return data.get_index(left) > data.get_index(right);
      }
    }
    if (heapType == HeapType::kMinHeap) {
      return left_perturbed_value < right_perturbed_value;
    } else {
      return left_perturbed_value > right_perturbed_value;
    }
  }

  __device__ void assign(int i, GumbelEntry const &entry) {
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

  __device__ void replace_root(GumbelEntry const &entry, int k) {
    data[0] = entry;
    push_root_down(k);
  }

  __device__ GumbelEntry const &root() {
    return data[0];
  }
};

template <HeapType heapType,
          PreferIndices preferIndices,
          template <typename>
          class Data,
          typename T>
__device__ IndexedHeap<heapType, preferIndices, Data, T>
    make_indexed_heap(typename Data<T>::GumbelEntry *data) {
  return IndexedHeap<heapType, preferIndices, Data, T>{Data<T>{data}};
}

__global__ void
    init_random_state_kernel(curandState *state, int batch_size, long rand) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    curand_init(rand, i, 0, &state[i]);
  }
}

// Unified log function for float
__device__ inline float unified_log(float x) {
  return logf(x);
}

// Unified log function for half
__device__ inline __half unified_log(__half x) {
  return hlog(x);
}

// heapGumbelTopK walks over [input, input+length) with `step_size` stride
// starting at `start_index`. It builds a top-`k` heap that is stored in
// `heap_entries` using `Accessor` to access elements in `heap_entries`. If
// sorted=true, the elements will be sorted at the end. NOTE that it applies
// Gumbel trick on `input`, which is, input -> log(input) - log(-log(U)), where
// U is a uniform random number in (0, 1).
template <typename T, template <typename> class Data = LinearData>
__device__ void heapGumbelTopK(curandState state,
                               T const *__restrict__ input,
                               int length,
                               int k,
                               GumbelEntry<T> *__restrict__ heap_entries,
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
    T value = unified_log(input[index]);
    T perturbed_value =
        value - unified_log(-unified_log((T)curand_uniform(&state)));
    heap.assign(slot, {index, value, perturbed_value});
  }

  heap.build(k);

  // Now iterate over the remaining items.
  // If an item is smaller than the min element, it is not amongst the top k.
  // Otherwise, replace the min element with it and push upwards.
  for (int index = heap_end_index; index < length; index += step_size) {
    // We prefer elements with lower indices. This is given here.
    // Later elements automatically have higher indices, so can be discarded.
    T value = unified_log(input[index]);
    T perturbed_value =
        value - unified_log(-unified_log((T)curand_uniform(&state)));
    if (perturbed_value > heap.root().perturbed_value) {
      // This element should replace the min.
      heap.replace_root({index, value, perturbed_value}, k);
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
// The overall top k elements are written to `top_k_values` and
// `top_k_perturbed_values`, and their indices to `top_k_indices`. `top_k_heap`
// is used as temporary storage for the merge heap.
template <typename T>
__device__ void mergeShards(int num_shards,
                            int k,
                            GumbelEntry<T> *__restrict__ entries,
                            GumbelEntry<T> *__restrict__ top_k_heap,
                            float *top_k_values,
                            float *top_k_perturbed_values,
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
      min_heap.assign(
          slot, {slot, entries[slot].value, entries[slot].perturbed_value});
    }
    min_heap.build(heap_size);

    // Now perform top k with the remaining shards (if num_shards > heap_size).
    for (int shard = heap_size; shard < num_shards; shard++) {
      auto const entry = entries[shard];
      auto const root = min_heap.root();
      if (entry.perturbed_value < root.perturbed_value) {
        continue;
      }
      if (entry.perturbed_value == root.perturbed_value &&
          entry.index > entries[root.index].index) {
        continue;
      }
      // This element should replace the min.
      min_heap.replace_root({shard, entry.value, entry.perturbed_value},
                            heap_size);
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
      GumbelEntry<T> const &max_element = max_heap.root();
      int shard_index = max_element.index;
      top_k_indices[rank] = entries[shard_index].index;
      if (speculative_decoding) {
        assert(top_k_values != nullptr);
        top_k_values[rank] = static_cast<float>(max_element.value);
        top_k_perturbed_values[rank] =
            static_cast<float>(max_element.perturbed_value);
      }
      int next_shard_index = shard_index + num_shards;
      // For rank < k-1, each top k heap still contains at least 1 element,
      // so we can draw a replacement.
      max_heap.replace_root({next_shard_index,
                             entries[next_shard_index].value,
                             entries[next_shard_index].perturbed_value},
                            heap_size);
    }

    // rank == last_k.
    GumbelEntry<T> const &max_element = max_heap.root();
    int shard_index = max_element.index;
    top_k_indices[last_k] = entries[shard_index].index;
    if (speculative_decoding) {
      assert(top_k_values != nullptr);
      top_k_values[last_k] = static_cast<float>(max_element.value);
      top_k_perturbed_values[last_k] =
          static_cast<float>(max_element.perturbed_value);
    }
  }
}

template <typename T>
__global__ void
    gumbel_topk_forward_kernel(curandState *state,
                               T const *__restrict__ input,
                               size_t shared_memory_size,
                               int length,
                               int k,
                               bool sorted,
                               float *__restrict__ log_probs_ptr,
                               float *__restrict__ perturbed_log_probs_ptr,
                               int *__restrict__ indices,
                               bool speculative_decoding) {
  __shared__ char shared_memory[48 << 10]; // block-wise shared memory
  int const batch_index = blockIdx.x;
  T const *batch_input = input + batch_index * length;
  int const thread_index = threadIdx.x;
  int const thread_count = blockDim.x;
  GumbelEntry<T> *shared_entries = (GumbelEntry<T> *)shared_memory;
  heapGumbelTopK<T, StridedData>(
      state[thread_index + batch_index * thread_count],
      batch_input,
      length,
      k,
      shared_entries,
      true,
      thread_index,
      thread_count);
  __syncthreads();
  if (thread_index == 0) {
    int const offset = batch_index * k;
    auto batch_log_probs_ptr = log_probs_ptr + offset;
    auto batch_perturbed_log_probs_ptr = perturbed_log_probs_ptr + offset;
    auto batch_indices = indices + offset;
    GumbelEntry<T> *top_k_heap = shared_entries + thread_count * k;
    mergeShards(thread_count,
                k,
                shared_entries,
                top_k_heap,
                batch_log_probs_ptr,
                batch_perturbed_log_probs_ptr,
                batch_indices,
                speculative_decoding);
  }
}

/*static*/
template <typename DT>
void GumbelTopK::forward_kernel(
    GumbelTopKMeta const *m,
    DT const *input_ptr,
    float *log_probs_ptr,
    float *perturbed_log_probs_ptr,
    int *indices_ptr,
    size_t batch_size,
    int length,
    int k,
    bool sorted,
    /* Reserved: BatchConfig Updated */ BatchConfig const *bc,
    cudaStream_t stream) {
  // Adopted from TensorFlow's ArgTopK implementation
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/topk_op_gpu.h
  int num_shards = 0;
  {
    constexpr auto shared_memory_size = 48 << 10;
    auto const heap_size = k * sizeof(GumbelEntry<DT>);
    // shared_memory_size = (num_shards + 1) * heap_size <=>
    num_shards = shared_memory_size / heap_size - 1;
    assert(num_shards > 0);
    if (num_shards > CUDA_NUM_THREADS) {
      num_shards = CUDA_NUM_THREADS;
    }
  }
  // We are limited by the amount of shared memory we have per block.
  size_t shared_memory_size = (num_shards + 1) * k * sizeof(GumbelEntry<DT>);
  // size_t num_blocks = (batch_size + num_shards - 1) / num_shards;
  size_t num_blocks = batch_size;

  // all requests share the same number of branches
  if (m->speculative_decoding) {
    assert(bc->num_active_requests() >= 0);
    assert(num_shards >= (size_t)BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES);
    num_shards = k;

    int state_length = batch_size * num_shards;
    init_random_state_kernel<<<GET_BLOCKS(state_length),
                               min((int)CUDA_NUM_THREADS, state_length),
                               0,
                               stream>>>(m->state, state_length, rand());

    gumbel_topk_forward_kernel<<<num_blocks, num_shards, 0, stream>>>(
        m->state,
        input_ptr,
        shared_memory_size,
        length,
        BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES,
        sorted,
        log_probs_ptr,
        perturbed_log_probs_ptr,
        indices_ptr,
        m->speculative_decoding);
  } else {
    assert(num_shards >= (size_t)k);
    num_shards = k;

    int state_length = batch_size * num_shards;
    init_random_state_kernel<<<GET_BLOCKS(state_length),
                               min((int)CUDA_NUM_THREADS, state_length),
                               0,
                               stream>>>(m->state, state_length, rand());

    gumbel_topk_forward_kernel<<<num_blocks, num_shards, 0, stream>>>(
        m->state,
        input_ptr,
        shared_memory_size,
        length,
        k,
        sorted,
        nullptr,
        nullptr,
        indices_ptr,
        false);
  }
}

/*static*/
void GumbelTopK::forward_kernel_wrapper(
    GumbelTopKMeta const *m,
    GenericTensorAccessorR const &input,
    // float *output_ptr,
    GenericTensorAccessorW const &log_probs,
    GenericTensorAccessorW const &perturbed_log_probs,
    GenericTensorAccessorW const &indices,
    int batch_size,
    BatchConfig const *bc) {
  cudaStream_t stream;
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

  // batch_size = input.domain.get_volume() / length;
  // assert(indices.domain.get_volume() / k == batch_size);
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  if (input.data_type == DT_HALF) {
    GumbelTopK::forward_kernel(
        m,
        input.get_half_ptr(),
        m->speculative_decoding ? log_probs.get_float_ptr() : nullptr,
        m->speculative_decoding ? perturbed_log_probs.get_float_ptr() : nullptr,
        indices.get_int32_ptr(),
        batch_size,
        length,
        k,
        m->sorted,
        m->speculative_decoding ? bc : nullptr,
        stream);
  } else if (input.data_type == DT_FLOAT) {
    GumbelTopK::forward_kernel(
        m,
        input.get_float_ptr(),
        m->speculative_decoding ? log_probs.get_float_ptr() : nullptr,
        m->speculative_decoding ? perturbed_log_probs.get_float_ptr() : nullptr,
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
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[GumbelTopK] forward time = %.2lfms\n", elapsed);
  }
}

GumbelTopKMeta::GumbelTopKMeta(FFHandler handler,
                               Op const *op,
                               MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, op) {
  state_max_length =
      BatchConfig::MAX_NUM_TOKENS *
      max(BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES, CUDA_NUM_THREADS);
  gpu_mem_allocator.create_legion_instance(
      reserveInst, sizeof(curandState) * state_max_length);
  state = gpu_mem_allocator.allocate_instance<curandState>(state_max_length);
}

GumbelTopKMeta::~GumbelTopKMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}
}; // namespace FlexFlow

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

#ifndef _FLEXFLOW_CONFIG_H_
#define _FLEXFLOW_CONFIG_H_
#include "ffconst.h"
#include "flexflow/batch_config.h"
#include "legion.h"
#include <cstddef>
#include <cstring>
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <cublas_v2.h>
#include <cudnn.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hipblas/hipblas.h>
#include <miopen/miopen.h>
#else
#error "Unknown device"
#endif
#include "tl/optional.hpp"
#ifdef FF_USE_NCCL
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <nccl.h>
#else
#include <rccl/rccl.h>
#endif
#endif

namespace FlexFlow {

// ========================================================
// Define Runtime Constants
// ========================================================
#define MAX_NUM_INPUTS 2048
#define MAX_NUM_WEIGHTS 2048
#define MAX_NUM_OUTPUTS 2048
#define MAX_NUM_FUSED_OPERATORS 2048
#define MAX_NUM_FUSED_TENSORS 2048
#define MAX_NUM_WORKERS 1024
#define MAX_FILENAME 200
#define MAX_OPNAME 128
#define MAX_NUM_TRANSFORMER_LAYERS 100
// DataLoader
#define MAX_SAMPLES_PER_LOAD 64
#define MAX_FILE_LENGTH 128
// Pre-assigned const flags
#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

#ifdef FF_USE_NCCL
constexpr ParameterSyncType CHOSEN_SYNC_TYPE = ParameterSyncType::NCCL;
#else
constexpr ParameterSyncType CHOSEN_SYNC_TYPE = ParameterSyncType::PS;
#endif

class FFConfig;

constexpr uint32_t kPagesize = 64;
#define DISPATCH_HEADDIM(head_dim, HEAD_DIM, ...)                              \
  switch (head_dim) {                                                          \
    case 64: {                                                                 \
      constexpr size_t HEAD_DIM = 64;                                          \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 128: {                                                                \
      constexpr size_t HEAD_DIM = 128;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 256: {                                                                \
      constexpr size_t HEAD_DIM = 256;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      std::ostringstream err_msg;                                              \
      err_msg << "Unsupported head_dim: " << head_dim;                         \
      throw std::invalid_argument(err_msg.str());                              \
    }                                                                          \
  }

class AttentionMetaData {
public:
  AttentionMetaData() {
    num_q_heads_ = 0;
    num_kv_heads_ = 0;
    head_dim_ = 0;
    q_indptr = nullptr;
    kv_indptr = nullptr;
    kv_indices = nullptr;
    kv_last_page_len = nullptr;
    qk_indptr = nullptr;
    custom_mask = nullptr;
    workspace = nullptr;
    workspace_size = 0;
    float_workspace = nullptr;
    float_workspace_size = 0;
    int_workspace = nullptr;
    int_workspace_size = 0;
    mem_size_ = 0;
    enabled_ = false;
  }
  AttentionMetaData(AttentionMetaData const &rhs) {
    num_q_heads_ = rhs.num_q_heads_;
    num_kv_heads_ = rhs.num_kv_heads_;
    head_dim_ = rhs.head_dim_;
    q_indptr = rhs.q_indptr;
    kv_indptr = rhs.kv_indptr;
    kv_indices = rhs.kv_indices;
    kv_last_page_len = rhs.kv_last_page_len;
    qk_indptr = rhs.qk_indptr;
    custom_mask = rhs.custom_mask;
    workspace = rhs.workspace;
    workspace_size = rhs.workspace_size;
    float_workspace = rhs.float_workspace;
    float_workspace_size = rhs.float_workspace_size;
    int_workspace = rhs.int_workspace;
    int_workspace_size = rhs.int_workspace_size;
    mem_size_ = rhs.mem_size_;
    enabled_ = rhs.enabled_;
    decode_handler_collections = rhs.decode_handler_collections;
    prompt_handler_collections = rhs.prompt_handler_collections;
  }

  size_t mem_size() {
    if (mem_size_ > 0) {
      return mem_size_;
    }
    size_t batch_size = BatchConfig::max_requests_per_batch();
    size_t max_num_pages =
        (BatchConfig::max_spec_tree_token_num() +
         BatchConfig::max_sequence_length() + kPagesize - 1) /
        kPagesize;
    size_t indices_size = std::max(
        (batch_size + 1) * 4 + max_num_pages * batch_size, 1ul * 1024 * 1024);
    size_t custom_mask_size = BatchConfig::max_requests_per_batch() *
                              ((BatchConfig::max_spec_tree_token_num() *
                                    (BatchConfig::max_spec_tree_token_num() +
                                     BatchConfig::max_sequence_length()) +
                                7) /
                               8);

    float_workspace_size = 128 * 1024 * 1024; // 128 MB
    int_workspace_size = 8 * 1024 * 1024;    // 8 MB
    workspace_size = float_workspace_size + int_workspace_size; // float + int workspace

    mem_size_ = sizeof(int32_t) * indices_size +
                sizeof(uint8_t) * custom_mask_size +
                workspace_size * BatchConfig::max_requests_per_batch();
    return mem_size_;
  }

  void assign_address(void *ptr, int size) {
    if (ptr == nullptr) {
      q_indptr = nullptr;
      kv_indptr = nullptr;
      kv_indices = nullptr;
      kv_last_page_len = nullptr;
      qk_indptr = nullptr;
      custom_mask = nullptr;
      workspace = nullptr;
      float_workspace = nullptr;
      int_workspace = nullptr;
      return;
    }
    assert(size >= mem_size() &&
           "Insufficient memory size for attention metadata");
    size_t batch_size = BatchConfig::max_requests_per_batch();
    size_t max_num_pages =
        (BatchConfig::max_spec_tree_token_num() +
         BatchConfig::max_sequence_length() + kPagesize - 1) /
        kPagesize;
    size_t indices_size = std::max(
        (batch_size + 1) * 4 + max_num_pages * batch_size, 1ul * 1024 * 1024);
    size_t custom_mask_size = BatchConfig::max_requests_per_batch() *
                              ((BatchConfig::max_spec_tree_token_num() *
                                    (BatchConfig::max_spec_tree_token_num() +
                                     BatchConfig::max_sequence_length()) +
                                7) /
                               8);

    q_indptr = static_cast<int32_t *>(ptr);
    kv_indptr = q_indptr + batch_size + 1;
    kv_indices = kv_indptr + batch_size + 1;
    kv_last_page_len = kv_indices + max_num_pages * batch_size;
    qk_indptr = kv_last_page_len + batch_size + 1;
    custom_mask = static_cast<uint8_t *>(ptr) + sizeof(int32_t) * indices_size;
    workspace = static_cast<void *>(static_cast<uint8_t *>(ptr) +
                                    sizeof(int32_t) * indices_size +
                                    sizeof(uint8_t) * custom_mask_size);
    float_workspace = workspace;
    int_workspace = static_cast<void *>(static_cast<uint8_t *>(workspace) +
                                        float_workspace_size);
  }

  void set_num_q_heads(uint32_t const num_q_heads) {
    num_q_heads_ = num_q_heads;
  }
  void set_num_kv_heads(uint32_t const num_kv_heads) {
    num_kv_heads_ = num_kv_heads;
  }
  void set_head_dim(uint32_t const head_dim) {
    head_dim_ = head_dim;
  }
  uint32_t num_q_heads() const {
    return num_q_heads_;
  }
  uint32_t num_kv_heads() const {
    return num_kv_heads_;
  }
  uint32_t head_dim() const {
    return head_dim_;
  }

  void set_enabled(bool const enabled) {
    enabled_ = enabled;
  }
  bool enabled() const {
    return enabled_;
  }

  uint32_t num_q_heads_;
  uint32_t num_kv_heads_;
  uint32_t head_dim_;

  int32_t *q_indptr;
  int32_t *kv_indptr;
  int32_t *kv_indices;
  int32_t *kv_last_page_len;
  int32_t *qk_indptr;
  uint8_t *custom_mask;
  void *workspace;
  size_t workspace_size;
  void * float_workspace;
  size_t float_workspace_size;
  void * int_workspace;
  size_t int_workspace_size;

  size_t mem_size_;

  // batchsize -> handler
  bool enabled_;
  std::unordered_map<int, void *> decode_handler_collections;
  std::unordered_map<int, void *> prompt_handler_collections;
};

struct FFHandler {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnHandle_t dnn;
  cublasHandle_t blas;
#else
  miopenHandle_t dnn;
  hipblasHandle_t blas;
#endif
  void *workSpace;
  size_t workSpaceSize;
  void *batch_config_metadata;
  AttentionMetaData *incr_attention_metadata;
  AttentionMetaData *tree_search_attention_metadata;
  AttentionMetaData *tree_verify_attention_metadata;

  size_t batch_config_metadata_size =
      sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo) +
      sizeof(BatchConfig::request_available) + sizeof(BatchConfig::causalMask) +
      sizeof(BatchConfig::committed_tokens) + sizeof(int);

  void *offload_reserve_space;
  size_t offload_reserve_space_size;
  DataType quantization_type;
  bool allowTensorOpMathConversion;
#ifdef FF_USE_NCCL
  ncclComm_t ncclComm;
  int num_devices;
  int device_id;
#endif
};

struct FFInitInfo {
  size_t workSpaceSize;
  size_t offload_reserve_space_size;
  DataType quantization_type;
  bool allowTensorOpMathConversion;
  // int myRank, allRanks;
};

// bool load_strategies_from_file(const std::string& filename,
//          std::map<Legion::MappingTagID, ParallelConfig>& strategies);

// bool save_strategies_to_file(const std::string& filename,
//                              const std::map<std::string, ParallelConfig>&
//                              strategies);

class FFConfig {
public:
  enum PreservedIDs {
    InvalidID = 0,
    DataParallelism_GPU = 1,
    // DataParallelism_GPU_2D = 2,
    // DataParallelism_GPU_3D = 3,
    // DataParallelism_GPU_4D = 4,
    // DataParallelism_GPU_5D = 5,
    DataParallelism_CPU = 11,
    // DataParallelism_CPU_2D = 12,
    // DataParallelism_CPU_3D = 13,
    // DataParallelism_CPU_4D = 14,
    // DataParallelism_CPU_5D = 15,
  };

  FFConfig();
  // bool load_strategy_file(std::string filename);
  // bool save_strategy_file(std::string filename);
  void parse_args(char **argv, int argc);
  static Legion::MappingTagID get_hash_id(std::string const &pcname);
  // bool find_parallel_config(int ndims,
  //                           const std::string& pcname,
  //                           ParallelConfig& config) const;
public:
  int epochs, batchSize, printFreq;
  // int inputHeight, inputWidth;
  int numNodes, cpusPerNode, workersPerNode;
  float device_mem; // The device (GPU) memory threshold; given by -ll:fsize
  float learningRate, weightDecay;
  size_t workSpaceSize;
  Legion::Context lg_ctx;
  Legion::Runtime *lg_hlr;
  Legion::IndexSpaceT<1> all_gpu_task_is;
  // Legion::FieldSpace field_space;
  bool benchmarking, profiling, perform_fusion;
  bool inference_debugging;
  size_t simulator_work_space_size;
  size_t search_budget;
  float search_alpha;
  bool search_overlap_backward_update;
  CompMode computationMode;
  bool cpu_offload;
  size_t offload_reserve_space_size;
  DataType quantization_type;
  // Control parallelizable dimensions
  bool only_data_parallel;
  bool enable_sample_parallel;
  bool enable_parameter_parallel;
  bool enable_attribute_parallel;
  bool enable_inplace_optimizations;
  // Control parallelism degrees in inference
  int data_parallelism_degree;
  int tensor_parallelism_degree;
  int pipeline_parallelism_degree;
  // Control Tensor Op Math Conversion
  bool allow_tensor_op_math_conversion;
  std::string dataset_path;
  std::string import_strategy_file;
  std::string export_strategy_file;
  std::string export_strategy_task_graph_file;
  std::string export_strategy_computation_graph_file;
  bool include_costs_dot_graph;
  tl::optional<std::string> substitution_json_path = tl::nullopt;
  // We use MappingTagID as the key since we will pass the tag to the mapper
  // std::map<Legion::MappingTagID, ParallelConfig> strategies;
  int machine_model_version;
  std::string machine_model_file;
  int simulator_segment_size;
  int simulator_max_num_segments;
  bool enable_propagation;
  tl::optional<int> search_num_nodes = tl::nullopt;
  tl::optional<int> search_num_workers = tl::nullopt;
  int base_optimize_threshold;
  bool enable_control_replication;
  int python_data_loader_type;
  bool perform_memory_search{false};
};

class FFIterationConfig {
public:
  FFIterationConfig();
  void reset();
  int seq_length;
};

enum FieldIDs {
  FID_DATA,
};

}; // namespace FlexFlow

#endif //_FLEXFLOW_CONFIG_H_

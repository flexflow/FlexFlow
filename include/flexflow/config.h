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
#include "legion.h"
#include <cstring>
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <cublas_v2.h>
#include <cudnn.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hipblas.h>
#include <miopen/miopen.h>
#else
#error "Unknown device"
#endif
#include "tl/optional.hpp"
#ifdef FF_USE_NCCL
#include <nccl.h>
#endif

namespace FlexFlow {

// ========================================================
// Define Runtime Constants
// ========================================================
#define MAX_NUM_INPUTS 256
#define MAX_NUM_WEIGHTS 64
#define MAX_NUM_OUTPUTS 256
#define MAX_NUM_FUSED_OPERATORS 64
#define MAX_NUM_FUSED_TENSORS 64
#define MAX_NUM_WORKERS 1024
#define MAX_FILENAME 200
#define MAX_OPNAME 128
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
  bool allowTensorOpMathConversion;
#ifdef FF_USE_NCCL
  ncclComm_t ncclComm;
#endif
};

struct FFInitInfo {
  size_t workSpaceSize;
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
  Legion::FieldSpace field_space;
  bool syntheticInput, profiling, perform_fusion;
  size_t simulator_work_space_size;
  size_t search_budget;
  float search_alpha;
  bool search_overlap_backward_update;
  CompMode computationMode;
  // Control parallelizable dimensions
  bool only_data_parallel;
  bool enable_sample_parallel;
  bool enable_parameter_parallel;
  bool enable_attribute_parallel;
  bool enable_inplace_optimizations;
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

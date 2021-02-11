/* Copyright 2018 Stanford
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
#include <cstring>
#include "legion.h"
#include "ffconst.h"
#include <cudnn.h>
#include <cublas_v2.h>
#ifdef FF_USE_NCCL
#include <nccl.h>
#endif

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
#define MAX_OPNAME 64
// DataLoader
#define MAX_SAMPLES_PER_LOAD 64
#define MAX_FILE_LENGTH 128
// Pre-assigned const flags
#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

using namespace Legion;

struct ParallelConfig {
  enum DeviceType {
    GPU = 0,
    CPU = 1,
  };
  bool operator==(const ParallelConfig &rhs) const
  {
    if (nDims != rhs.nDims) return false;
    if (device_type != rhs.device_type) return false;
    for (int i = 0; i < nDims; i++)
      if (dim[i] != rhs.dim[i])
        return false;
    for (int i = 0; i < num_parts(); i++)
      if (device_ids[i] != rhs.device_ids[i])
        return false;
    return true;
  }
  int num_parts() const;
  bool is_data_parallel() const;
  DeviceType device_type;
  int nDims, dim[MAX_TENSOR_DIM];
  int device_ids[MAX_NUM_WORKERS];
#ifdef FF_USE_NCCL
  ncclComm_t nccl_comms[MAX_NUM_WORKERS];
#endif
};

struct FFHandler {
  cudnnHandle_t dnn;
  cublasHandle_t blas;
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
  //int myRank, allRanks;
};

bool load_strategies_from_file(const std::string& filename,
         std::map<MappingTagID, ParallelConfig>& strategies);

bool save_strategies_to_file(const std::string& filename,
                             const std::map<std::string, ParallelConfig>& strategies);

class FFConfig {
public:
  enum PreservedIDs{
    InvalidID = 0,
    DataParallelism_GPU_1D = 1,
    DataParallelism_GPU_2D = 2,
    DataParallelism_GPU_3D = 3,
    DataParallelism_GPU_4D = 4,
    DataParallelism_GPU_5D = 5,
    DataParallelism_CPU_1D = 11,
    DataParallelism_CPU_2D = 12,
    DataParallelism_CPU_3D = 13,
    DataParallelism_CPU_4D = 14,
    DataParallelism_CPU_5D = 15,
  };

  FFConfig();
  //bool load_strategy_file(std::string filename);
  //bool save_strategy_file(std::string filename);
  void parse_args(char** argv, int argc);
  static MappingTagID get_hash_id(const std::string& pcname);
  bool find_parallel_config(int ndims,
                            const std::string& pcname,
                            ParallelConfig& config) const;
public:
  int epochs, batchSize, iterations, printFreq;
  //int inputHeight, inputWidth;
  int numNodes, cpusPerNode, workersPerNode;
  float learningRate, weightDecay;
  size_t workSpaceSize;
  Context lg_ctx;
  Runtime* lg_hlr;
  FieldSpace field_space;
  bool syntheticInput, profiling, perform_fusion;
  size_t simulator_work_space_size;
  size_t search_budget;
  float search_alpha;
  bool search_overlap_backward_update;
  CompMode computationMode;
  std::string export_strategy_task_graph_file;
  //Control parallelizable dimensions
  bool enable_sample_parallel;
  bool enable_parameter_parallel;
  bool enable_attribute_parallel;
  //Control Tensor Op Math Conversion
  bool allow_tensor_op_math_conversion;
  std::string dataset_path;
  std::string import_strategy_file;
  std::string export_strategy_file;
  // We use MappingTagID as the key since we will pass the tag to the mapper
  std::map<MappingTagID, ParallelConfig> strategies;
  int machine_model_version;
  std::string machine_model_file;
  int simulator_segment_size;
  int simulator_max_num_segments;
};

class FFIterationConfig {
public:
  FFIterationConfig();
  void reset();
  int seq_length;
};

struct ParaConfigCompare {
  bool operator()(const ParallelConfig& a, const ParallelConfig& b) const {
    if (a.nDims != b.nDims)
      return a.nDims < b.nDims;
    for (int i = 0; i < a.nDims; i++)
      if (a.dim[i] != b.dim[i])
        return a.dim[i] < b.dim[i];
    return false;
  }
};
#endif//_FLEXFLOW_CONFIG_H_

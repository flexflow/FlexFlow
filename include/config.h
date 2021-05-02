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

//using namespace Legion;

#ifdef FF_USE_NCCL
constexpr ParameterSyncType CHOSEN_SYNC_TYPE = ParameterSyncType::NCCL;
#else 
constexpr ParameterSyncType CHOSEN_SYNC_TYPE = ParameterSyncType::PS;
#endif

struct MachineView {
  static const MachineView NO_VIEW;
  MachineView();
  inline int get_device_id(const Legion::DomainPoint& p) const
  {
    assert(p.get_dim() == ndims);
    int idx = start_device_id;
    for (int i = 0; i < ndims; i++)
      idx += p[i] * stride[i];
    return idx;
  }
  inline bool operator==(const MachineView& rhs) const
  {
    if (device_type != rhs.device_type) return false;
    if (ndims != rhs.ndims) return false;
    if (start_device_id != rhs.start_device_id) return false;
    for (int i = 0; i < ndims; i++) {
      if (dim[i] != rhs.dim[i]) return false;
      if (stride[i] != rhs.stride[i]) return false;
    }
    return true;
  }
  inline bool operator!=(const MachineView& rhs) const
  {
    if (device_type != rhs.device_type) return true;
    if (ndims != rhs.ndims) return true;
    if (start_device_id != rhs.start_device_id) return true;
    for (int i = 0; i < ndims; i++) {
      if (dim[i] != rhs.dim[i]) return true;
      if (stride[i] != rhs.stride[i]) return true;
    }
    return false;
  }
  size_t hash() const;
  size_t num_parts() const;
  enum DeviceType {
    GPU = 0,
    CPU = 1,
  };
  DeviceType device_type;
  int ndims, start_device_id, dim[MAX_TENSOR_DIM], stride[MAX_TENSOR_DIM];
  std::vector<int> device_ids() const;
  std::vector<Legion::Domain> domains() const;
  Legion::Domain domain_from_point(Legion::DomainPoint const &) const;
};

struct MachineResource {
  bool is_valid_machine_view(const MachineView& view) const;
  size_t hash() const;
  int num_nodes;
  int all_gpus_per_node, available_gpus_per_node;
  int all_cpus_per_node, available_cpus_per_node;
  int start_gpu_id, start_cpu_id;
};

struct ParallelOpInfo {
  OperatorType op_type;
  int parallel_dim;
  int parallel_degree;
};

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
  ParallelConfig change_data_parallel_dimensionality(int new_dimensionality) const;
  DeviceType device_type;
  int nDims, dim[MAX_TENSOR_DIM];
  int device_ids[MAX_NUM_WORKERS];
  Legion::Domain domains[MAX_TENSOR_DIM]; // for use in view to pc conversion
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

//bool load_strategies_from_file(const std::string& filename,
//         std::map<Legion::MappingTagID, ParallelConfig>& strategies);

//bool save_strategies_to_file(const std::string& filename,
//                             const std::map<std::string, ParallelConfig>& strategies);

class FFConfig {
public:
  enum PreservedIDs{
    InvalidID = 0,
    DataParallelism_GPU = 1,
    //DataParallelism_GPU_2D = 2,
    //DataParallelism_GPU_3D = 3,
    //DataParallelism_GPU_4D = 4,
    //DataParallelism_GPU_5D = 5,
    DataParallelism_CPU = 11,
    //DataParallelism_CPU_2D = 12,
    //DataParallelism_CPU_3D = 13,
    //DataParallelism_CPU_4D = 14,
    //DataParallelism_CPU_5D = 15,
  };

  FFConfig();
  //bool load_strategy_file(std::string filename);
  //bool save_strategy_file(std::string filename);
  void parse_args(char** argv, int argc);
  static Legion::MappingTagID get_hash_id(const std::string& pcname);
  //bool find_parallel_config(int ndims,
  //                          const std::string& pcname,
  //                          ParallelConfig& config) const;
public:
  int epochs, batchSize, printFreq;
  //int inputHeight, inputWidth;
  int numNodes, cpusPerNode, workersPerNode;
  float learningRate, weightDecay;
  size_t workSpaceSize;
  Legion::Context lg_ctx;
  Legion::Runtime* lg_hlr;
  Legion::FieldSpace field_space;
  bool syntheticInput, profiling, perform_fusion;
  size_t simulator_work_space_size;
  size_t search_budget;
  float search_alpha;
  bool search_overlap_backward_update;
  CompMode computationMode;
  //Control parallelizable dimensions
  bool only_data_parallel;
  bool enable_sample_parallel;
  bool enable_parameter_parallel;
  bool enable_attribute_parallel;
  bool enable_inplace_optimizations;
  //Control Tensor Op Math Conversion
  bool allow_tensor_op_math_conversion;
  std::string dataset_path;
  std::string import_strategy_file;
  std::string export_strategy_file;
  std::string export_strategy_task_graph_file;
  std::string export_strategy_computation_graph_file;
  std::string search_curve_file;
  int search_curve_interval;
  // We use MappingTagID as the key since we will pass the tag to the mapper
  //std::map<Legion::MappingTagID, ParallelConfig> strategies;
  int machine_model_version;
  std::string machine_model_file;
  int simulator_segment_size;
  int simulator_max_num_segments;
  bool enable_propagation;
};

class FFIterationConfig {
public:
  FFIterationConfig();
  void reset();
  int seq_length;
};

struct MachineViewDimCompare {
  bool operator()(const MachineView& a, const MachineView& b) const {
    if (a.ndims != b.ndims)
      return a.ndims < b.ndims;
    for (int i = 0; i < a.ndims; i++)
      if (a.dim[i] != b.dim[i])
        return a.dim[i] < b.dim[i];
    return false;
  }
};
#endif//_FLEXFLOW_CONFIG_H_

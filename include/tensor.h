/* Copyright 2020 Facebook
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

#ifndef _FLEXFLOW_TENSOR_H_
#define _FLEXFLOW_TENSOR_H_
#include "legion.h"
#include "config.h"
#include "ffconst.h"

//using namespace Legion;

class Op;
class FFModel;
class Initializer;

struct ParallelDim {
  ParallelDim(): size(0), degree(1), parallel_idx(-1) {}
  int size;
  int degree;
  int parallel_idx;
};

struct TensorBase {
  TensorBase(void);
  //Tensor& operator=(const Tensor& rhs);
  //bool operator==(const Tensor& rhs) const;
  void inline_map(FFConfig &config);
  void inline_unmap(FFConfig &config);
  template<typename T>
  T* get_raw_ptr(FFConfig &config);
  void attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major);
  void detach_raw_ptr(FFConfig &config);
  bool get_input_sub_tensor(const ParallelConfig& pc,
                            TensorBase& tensor,
                            OperatorType type);
  bool get_output_sub_tensor(const ParallelConfig& pc,
                             TensorBase& tensor,
                             OperatorType type);
  size_t get_volume() const;
  Legion::Domain get_domain() const;
  bool check_valid() const;
  static bool update_parallel_ids(int numdim, ParallelDim* dims);
  template <typename T>
  bool set_tensor(const FFModel* model,
                   const std::vector<int>& dims,
                   const T* data);
  template <typename T>
  bool get_tensor(const FFModel* model,
                  T* data);
  int guid;
  int num_dims;
  //int adim[MAX_TENSOR_DIM];
  ParallelDim dims[MAX_TENSOR_DIM];
  DataType data_type;
  ParameterSyncType sync_type;
  Initializer* initializer;
  // Describes the ownership of this tensor
  const Op* owner_op;
  int owner_idx;
  bool create_gradients;
  // The following fields are initialized after model.compile
  Legion::IndexSpace parallel_is;
  Legion::LogicalRegion region, region_grad;
  Legion::LogicalPartition part, part_grad;
  Legion::PhysicalRegion physical_region;
};

typedef TensorBase* Tensor;
typedef TensorBase* Parameter;

/*
struct Parameter : TensorBase {
  template <typename T>
  bool set_weights(const FFModel* model,
                   const std::vector<int>& dims,
                   const T* data);
  template <typename T>
  bool get_weights(const FFModel* model,
                   T* data);
};
*/

#endif

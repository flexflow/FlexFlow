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

using namespace Legion;

class Op;
class FFModel;

struct Tensor {
  Tensor(void);
  Tensor& operator=(const Tensor& rhs);
  void inline_map(FFConfig &config);
  void inline_unmap(FFConfig &config);
  template<typename T>
  T* get_raw_ptr(FFConfig &config);
  void attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major);
  void detach_raw_ptr(FFConfig &config);
  bool get_input_sub_tensor(const ParallelConfig& pc,
                            Tensor& tensor,
                            OperatorType type);
  bool get_output_sub_tensor(const ParallelConfig& pc,
                             Tensor& tensor,
                             OperatorType type);
  size_t get_volume() const;
  Domain get_domain() const;
  template <typename T>
  bool set_tensor(const FFModel* model,
                   const std::vector<int>& dims,
                   const T* data,
                   ParameterSyncType comm_type);
  template <typename T>
  bool get_tensor(const FFModel* model,
                  T* data,
                  ParameterSyncType comm_type);
  int numDim, adim[MAX_TENSOR_DIM];
  DataType data_type;
  ParameterSyncType sync_type;
  // Describes the ownership of this tensor
  Op* owner_op;
  int owner_idx;
  // The following fields are initialized after model.compile
  LogicalRegion region, region_grad;
  LogicalPartition part, part_grad;
  PhysicalRegion physical_region;
};

struct Parameter : Tensor {
  template <typename T>
  bool set_weights(const FFModel* model,
                   const std::vector<int>& dims,
                   const T* data);
  template <typename T>
  bool get_weights(const FFModel* model,
                   T* data);
};

#endif

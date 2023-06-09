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

#ifndef __FLEXFLOW_DATALOADER_H__
#define __FLEXFLOW_DATALOADER_H__

#include "flexflow/model.h"

struct NetConfig {
  NetConfig(void);
  std::string dataset_path;
};

struct DLRMConfig {
  DLRMConfig(void);
  int sparse_feature_size, sigmoid_bot, sigmoid_top, embedding_bag_size;
  float loss_threshold;
  std::vector<int> embedding_size, mlp_bot, mlp_top;
  std::string arch_interaction_op, dataset_path;
};

class SingleDataLoader {
public:
  SingleDataLoader(FlexFlow::FFModel &ff,
                   FlexFlow::ParallelTensor input,
                   FlexFlow::ParallelTensor full_input_,
                   int num_samples_,
                   DataType datatype_);

  SingleDataLoader(FlexFlow::FFModel &ff,
                   FlexFlow::ParallelTensor input,
                   void *full_input_ptr,
                   int num_samples_,
                   DataType datatype_);

  void next_batch(FlexFlow::FFModel &);

  void reset(void);

  static void register_cpu_tasks(void);

  static void register_gpu_tasks(void);

  template <typename DT>
  static void load_input(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  // template<typename DT, int NDIM>
  // static void load_input_with_dim(
  //     const Legion::Task *task,
  //     const std::vector<Legion::PhysicalRegion> &regions,
  //     Legion::Context ctx,
  //     Legion::Runtime* runtime);
  template <typename DT>
  static void load_entire_dataset_from_numpy(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename DT, int NDIM>
  static void load_entire_dataset_from_numpy_with_dim(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename DT>
  static void index_load_entire_dataset_from_numpy(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename DT, int NDIM>
  static void index_load_entire_dataset_from_numpy_with_dim(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

private:
  template <int NDIM>
  void next_batch_xd_launcher(FlexFlow::FFModel &ff, int task_id);

  template <int NDIM>
  void index_loader_xd_launcher(FlexFlow::FFModel &ff,
                                int task_id,
                                void *full_input_ptr,
                                size_t size_per_sample);

public:
  int num_samples, next_index;
  DataType datatype;
  FlexFlow::ParallelTensor full_input, batch_input;
};

#define MAX_NUM_SAMPLES 4196
struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

struct IndexLoadArg {
  int num_samples;
  size_t size_per_sample;
  int idx;
  void *ptr;
};

#endif // __FLEXFLOW_DATALOADER_H__

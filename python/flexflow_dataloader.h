/* Copyright 2020 Stanford, Los Alamos National Laboratory
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

// TODO: remove data loaders except single data loader
class ImgDataLoader {
public:
  ImgDataLoader();
  static void load_label(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  void reset(void);

public:
  int num_samples, next_index;
  FlexFlow::ParallelTensor full_input, batch_input;
  FlexFlow::ParallelTensor full_label, batch_label;
};

class ImgDataLoader4D : public ImgDataLoader {
public:
  ImgDataLoader4D(FlexFlow::FFModel &ff,
                  FlexFlow::ParallelTensor input,
                  FlexFlow::ParallelTensor label,
                  FlexFlow::ParallelTensor full_input_,
                  FlexFlow::ParallelTensor full_label_,
                  int num_samples_);
  ImgDataLoader4D(FlexFlow::FFModel &ff,
                  const NetConfig &alexnet,
                  FlexFlow::ParallelTensor input,
                  FlexFlow::ParallelTensor label);
  static void load_input(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  static void
  load_entire_dataset(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context ctx,
                      Legion::Runtime *runtime);
  static void load_entire_dataset_from_numpy(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  void next_batch(FlexFlow::FFModel &);

private:
  size_t get_file_size(const std::string &filename);
};

class ImgDataLoader2D : public ImgDataLoader {
public:
  ImgDataLoader2D(FlexFlow::FFModel &ff,
                  FlexFlow::ParallelTensor input,
                  FlexFlow::ParallelTensor label,
                  FlexFlow::ParallelTensor full_input_,
                  FlexFlow::ParallelTensor full_label_,
                  int num_samples_);
  static void load_input(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  static void load_entire_dataset_from_numpy(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  void next_batch(FlexFlow::FFModel &);
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
  static void load_input(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
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
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename DT, int NDIM>
  static void load_entire_dataset_from_numpy_with_dim(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename DT>
  static void index_load_entire_dataset_from_numpy(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename DT, int NDIM>
  static void index_load_entire_dataset_from_numpy_with_dim(
      const Legion::Task *task,
      const std::vector<Legion::PhysicalRegion> &regions,
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

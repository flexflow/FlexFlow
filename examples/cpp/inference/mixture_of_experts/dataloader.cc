
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

#include "data_generator.h"
#include "flexflow/inference.h"
#include "moe.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Legion;
LegionRuntime::Logger::Category log_app("MoE");

DataLoader::DataLoader(FFModel &ff,
                       MoeConfig const &moeConfig,
                       DataGenerator &data_generator,
                       ParallelTensor input,
                       ParallelTensor label) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  int numdims = input->num_dims;
  int replica_idx = numdims - 1;
  int batch_idx = numdims - 1;
  num_samples = moeConfig.total_requests;

  // Create full input
  {
    batch_input = input;

    ParallelDim dims[numdims];
    for (int i = 0; i < numdims; i++) {
      dims[i].size = input->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = input->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == replica_idx || (!dims[i].is_replica_dim));
    }
    dims[1].size = num_samples;

    full_input =
        ff.create_parallel_tensor_legion_ordering(numdims, dims, DT_FLOAT);
    ff.map_tensor(full_input, NULL /*parallel_op*/);
  }

  // Create full label
  {
    assert(label->num_dims == numdims);
    batch_label = label;

    ParallelDim dims[numdims];
    for (int i = 0; i < numdims; i++) {
      dims[i].size = label->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = label->dims[i].is_replica_dim;
      // Assume only the last dim can be the replica dim
      assert(i == replica_idx || (!dims[i].is_replica_dim));
    }
    assert(dims[batch_idx].size == ff.config.batchSize);
    // replace batch size with number of samples
    dims[batch_idx].size = num_samples;

    full_label =
        ff.create_parallel_tensor_legion_ordering(numdims, dims, DT_INT32);
    ff.map_tensor(full_label, NULL /*parallel_op*/);
  }

  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  assert(full_input != nullptr && "full_input is nullptr");
  assert(full_label != nullptr && "full_label is nullptr");

  DataLoaderInput dataloader_input = {moeConfig, data_generator};
  DataLoaderInput const *ptr = &dataloader_input;

  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
                        TaskArgument(&ptr, sizeof(DataLoaderInput *)));
  // regions[0]: full_input
  launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_input->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(RegionRequirement(full_label->region,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_label->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);

  runtime->execute_task(ctx, launcher);
  reset();
  next_batch(ff);
}

// =================================================
//                    Load data
// =================================================

void read_cifar100(float *input_ptr, int *label_ptr) {
  std::ifstream file;
  file.open("train.bin", std::ios::in | std::ios::binary | std::ios::ate);
  if (!file) {
    std::cout << "Error opening CIFAR100 train data file" << std::endl;
    assert(false);
  }

  file.seekg(0, std::ios::beg);

  // each sample: <1 x coarse label><1 x fine label><3072 x pixel>
  for (std::size_t i = 0; i < NUM_SAMPLES; i++) {
    unsigned char temp = 0;
    file.read((char *)&temp, sizeof(temp)); // coarse label, skip
    file.read((char *)&temp, sizeof(temp));
    label_ptr[i] = temp;
    for (std::size_t j = 0; j < 3072; ++j) {
      file.read((char *)&temp, sizeof(temp));
      input_ptr[i * 3072 + j] = (float)temp / 255.0f;
    }
  }

  file.close();
}

int reverseInt(int i) {
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/* NOTE: Download files from http://yann.lecun.com/exdb/mnist/ and unpack to
the current working directory */
void read_mnist(float *input_ptr, int *label_ptr) {
  // read inputs
  std::ifstream input("train-images-idx3-ubyte", std::ios::binary);
  if (input.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    input.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    input.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    input.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    input.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    for (int i = 0; i < number_of_images; i++) {
      for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
          unsigned char temp = 0;
          input.read((char *)&temp, sizeof(temp));
          input_ptr[i * n_rows * n_cols + r * n_cols + c] =
              (float)temp / 255.0f;
        }
      }
    }
  } else {
    std::cout << "Error opening MNIST input data file" << std::endl;
    assert(false);
  }

  // read labels
  std::ifstream labels("train-labels-idx1-ubyte", std::ios::binary);
  if (labels.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    labels.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    labels.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    for (int i = 0; i < number_of_images; i++) {
      unsigned char temp = 0;
      labels.read((char *)&temp, sizeof(temp));
      label_ptr[i] = temp;
    }
  } else {
    std::cout << "Error opening MNIST label data file" << std::endl;
    assert(false);
  }
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  DataLoaderInput const *input_struct = *((DataLoaderInput **)task->args);

  MoeConfig const *conf = &input_struct->_moeConfig;
  DataGenerator *datagen = &input_struct->_data_generator;
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());

  // get input and label pointer
  float *input_ptr = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  int *label_ptr = helperGetTensorPointerWO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain label_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  int input_dims = input_domain.get_dim();
  for (int i = 0; i < input_dims; i++) {
    int input_dim = input_domain.hi()[i] - input_domain.lo()[i] + 1;
    int label_dim = label_domain.hi()[i] - label_domain.lo()[i] + 1;
    assert(i == 0 || input_dim == label_dim);
  }

  if (conf->dataset_path.length() == 0) {
    log_app.print("Input dataset path is empty, using random input samples\n");
    datagen->generate_requests(input_ptr, label_ptr, conf->num_labels);
  } else {
    // here, you can call `read_cifar100(input_ptr, label_ptr);` instead or load
    // another dataset using the dataset_path from the MoeConfig object
    // read_mnist(input_ptr, label_ptr);
    // log_app.print("finish loading MNIST data\n");
  }
}

void DataLoader::next_batch(FFModel &ff, size_t received_requests) {
  if (received_requests == 0) {
    return;
  }
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load input
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_input->parallel_is);
    ArgumentMap argmap;
    int counter = 0;
    // current limitation of the dataloader: only the batch dimension can be
    // partitioned
    int input_dims = batch_input->num_dims;
    for (int i = 0; i < input_dims; i++) {
      if (i != input_dims - 2) {
        assert(batch_input->dims[i].degree == 1 &&
               "Dataloader only supports batch size partitions");
      }
    }
    int batch_size = batch_input->dims[input_dims - 2].size;
    int n_partitions = batch_input->dims[input_dims - 2].degree;
    assert(ff.config.batchSize % batch_size == 0);
    assert(batch_size % n_partitions == 0);
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      int requests_left = received_requests - counter;
      meta.num_samples = std::min(batch_size / n_partitions, requests_left);
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = next_index + counter;
        counter++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    assert(counter == received_requests);
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1,
                           batch_input->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_input->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_input->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_input->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_input->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_label->parallel_is);
    ArgumentMap argmap;
    int counter = 0;
    // current limitation of the dataloader: only the batch dimension can be
    // partitioned
    int label_dims = batch_label->num_dims;
    // assert(batch_label->dims[label_dims - 1].degree == 1);
    for (int i = 0; i < label_dims; i++) {
      assert(batch_label->dims[i].degree == 1 &&
             "Dataloader only supports batch size partitions");
    }
    int batch_size = batch_label->dims[label_dims - 2].size;
    int n_partitions = batch_label->dims[label_dims - 2].degree;
    assert(ff.config.batchSize % batch_size == 0);
    assert(batch_size % n_partitions == 0);
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      int requests_left = received_requests - counter;
      meta.num_samples = std::min(batch_size / n_partitions, requests_left);
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = next_index + counter;
        counter++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    assert(counter == received_requests);
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2,
                           batch_label->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_label->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(full_label->region,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_label->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_label->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_label->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += received_requests;
}

void DataLoader::reset() {
  next_index = 0;
}

void FlexFlow::register_custom_tasks() {
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Input Task");
  }
  // Load label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
}

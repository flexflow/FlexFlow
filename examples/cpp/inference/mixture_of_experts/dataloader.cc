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

#include "flexflow/inference.h"
#include "moe.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Legion;

DataLoader::DataLoader(FFModel &ff,
                       MoeConfig const &moeConfig,
                       DataGenerator &data_generator,
                       ParallelTensor input,
                       ParallelTensor label) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  int numdims = input->num_dims;
  int replica_idx = numdims - 1;
  int batch_idx = numdims - 2;
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
    assert(dims[batch_idx].size == ff.config.batchSize);
    dims[batch_idx].size = num_samples;

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
                        TaskArgument(ptr, sizeof(DataLoaderInput)));
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
}

// =================================================
//                    Load data
// =================================================

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  DataLoaderInput const input_struct = *((DataLoaderInput *)task->args);
  MoeConfig const &conf = input_struct._moeConfig;
  DataGenerator &datagen = input_struct._data_generator;
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

  if (conf.dataset_path.length() == 0) {
    printf("Input dataset path is empty, using random input samples\n");
    datagen.generate_requests(input_ptr, label_ptr, conf.num_labels);
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

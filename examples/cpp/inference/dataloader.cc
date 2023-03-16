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
#include "transformers.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Legion;

DataLoader::DataLoader(FFModel &ff,
                       InferenceConfig const &inferenceConfig,
                       DataGenerator &data_generator,
                       ParallelTensor input) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  int numdims = input->num_dims;
  int replica_idx = numdims - 1;
  int batch_idx = numdims - 2;
  num_samples = inferenceConfig.total_requests;
  max_sequence_length = data_generator.max_sequence_length;

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
    assert(dims[batch_idx].size == BatchConfig::MAX_NUM_TOKENS);
    dims[batch_idx].size = num_samples;

    full_input =
        ff.create_parallel_tensor_legion_ordering(numdims, dims, DT_FLOAT);
    ff.map_tensor(full_input, NULL /*parallel_op*/);
  }

  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  assert(full_input != nullptr && "full_input is nullptr");

  DataLoaderInput dataloader_input = {inferenceConfig, data_generator};
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

  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  DataLoaderInput const input_struct = *((DataLoaderInput *)task->args);
  InferenceConfig const &conf = input_struct._inferenceConfig;
  DataGenerator &datagen = input_struct._data_generator;
  assert(regions.size() == 1);
  assert(task->regions.size() == regions.size());

  // get input pointer
  float *input_ptr = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  int input_dims = input_domain.get_dim();
  for (int i = 0; i < input_dims; i++) {
    int input_dim = input_domain.hi()[i] - input_domain.lo()[i] + 1;
  }

  if (conf.dataset_path.length() == 0) {
    printf("Input dataset path is empty, using random input samples\n");
    datagen.generate_requests(input_ptr);
  } else {
    // Load specific dataset
  }
}

void DataLoader::next_batch(FFModel &ff, BatchConfig *bc) {
  size_t num_active_tokens = bc->num_active_tokens();
  if (num_active_tokens == 0) {
    return;
  }
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load input
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_input->parallel_is);
    ArgumentMap argmap;
    // No partitioning of the batch input token in inference mode
    int input_dims = batch_input->num_dims;
    for (int i = 0; i < input_dims; i++) {
      assert(batch_input->dims[i].degree == 1 &&
             "Dataloader does not support input token partitioning in "
             "inference mode");
    }
    int batch_size = batch_input->dims[input_dims - 2].size;
    assert(ff.config.batchSize == batch_size &&
           batch_size >= num_active_tokens);
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      meta.num_samples = num_active_tokens;
      int token_index = 0;
      for (int i = 0; i < bc->MAX_NUM_REQUESTS; i++) {
        if (bc->request_completed[i]) {
          continue;
        } else {
          for (int j = 0; j < bc->num_processing_tokens[i]; j++) {
            meta.guids[token_index] = bc->request_guid[i];
            meta.idxs[token_index] = bc->token_start_idx[i] + j;
            token_index++;
          }
        }
      }
      assert(token_index == num_active_tokens);
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
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
}

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

#include "dataloader.h"
#include "flexflow/inference.h"
#include "inference_config.h"

using namespace Legion;

DataLoader::DataLoader(FFModel &ff,
                       InferenceConfig const &inferenceConfig,
                       DataGenerator &data_generator,
                       std::vector<ParallelTensor> input) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  assert(input.size() > 0);
  int numdims = input[0]->num_dims;
  for (int i = 1; i < input.size(); i++) {
    assert(input[i]->num_dims == numdims);
    for (int j = 0; j < numdims; j++) {
      assert(input[i]->dims[j].size == input[0]->dims[j].size);
      assert(input[i]->dims[j].degree == input[0]->dims[j].degree);
      assert(input[i]->dims[j].parallel_idx == input[0]->dims[j].parallel_idx);
    }
  }

  int replica_idx = numdims - 1;
  int batch_idx = numdims - 2;
  num_samples = inferenceConfig.total_requests;

  // Create full input
  {
    batch_input = input;

    ParallelDim dims[numdims];
    for (int i = 0; i < numdims; i++) {
      dims[i].size = input[0]->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = input[0]->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == replica_idx || (!dims[i].is_replica_dim));
    }
    assert(dims[batch_idx].size == inferenceConfig.batch_size);
    dims[batch_idx].size = num_samples;

    full_input =
        ff.create_parallel_tensor_legion_ordering(numdims, dims, DT_INT32);
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
  int *input_ptr = helperGetTensorPointerWO<int>(
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
    printf("Loading dataset from %s\n", conf.dataset_path.c_str());
    assert(datagen.load_requests(input_ptr,
                                 conf.dataset_path,
                                 conf.token_to_generate,
                                 conf.arrival_info_path));
  }
}

void DataLoader::next_batch(FFModel &ff,
                            int bid,
                            BatchConfig *bc,
                            std::map<size_t, int> &batch_predictions,
                            MachineView const *mv) {
  size_t num_active_tokens = bc->num_active_tokens();
  if (num_active_tokens == 0) {
    return;
  }
  assert(bid < batch_input.size());
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load input
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_input[bid]->parallel_is);
    ArgumentMap argmap;
    // No partitioning of the batch input token in inference mode
    int input_dims = batch_input[bid]->num_dims;
    for (int i = 0; i < input_dims; i++) {
      assert(batch_input[bid]->dims[i].degree == 1 &&
             "Dataloader does not support input token partitioning in "
             "inference mode");
    }
    int batch_size = batch_input[bid]->dims[input_dims - 2].size;
    int seq_len = batch_input[bid]->dims[input_dims - 3].size;
    /* printf("ff.config.batchSize: %i, batch_size: %i, seq_len: %i,
       num_active_tokens: %i\n", ff.config.batchSize, batch_size, seq_len,
       num_active_tokens); */
    assert(ff.config.batchSize == batch_size &&
           batch_size * seq_len >= num_active_tokens);

    /* std::cout << "About to call next_batch function..." << std::endl;
    bc->print();
    std::cout << "batch_predictions: ";
    for (const auto& elem : batch_predictions){
        std::cout << elem.first << ":" << elem.second << ", ";
    } */
    DataLoaderNextBatchInput next_batch_input = {bc->token2ids,
                                                 batch_predictions};
    DataLoaderNextBatchInput const *ptr = &next_batch_input;
    size_t next_batch_input_sz = sizeof(next_batch_input);
    assert(ptr->prev_batch_preds.size() == batch_predictions.size());
    MachineView const *view = mv ? mv : &batch_input[bid]->machine_view;
    size_t machine_view_hash = view->hash();
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1,
                           batch_input[bid]->parallel_is,
                           TaskArgument(ptr, next_batch_input_sz),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           machine_view_hash);
    launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_input->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input[bid]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_input[bid]->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

void DataLoader::store_outputs(BatchConfig *bc,
                               InferenceResult const &ir,
                               std::map<size_t, int> &batch_predictions) {
  assert(bc->token2ids.num_samples == bc->num_active_tokens() &&
         bc->token2ids.num_samples <= bc->MAX_NUM_TOKENS);
  batch_predictions.clear();
  // bc->print();
  for (size_t i = 0; i < bc->token2ids.num_samples; i++) {
    if (i == bc->token2ids.num_samples - 1 ||
        bc->token2ids.guids[i] != bc->token2ids.guids[i + 1]) {
      assert(bc->token2ids.token_indexes[i].token_position ==
             bc->token_last_available_idx[bc->token2ids.token_indexes[i]
                                              .request_index]);
      if (outputs.find(bc->token2ids.guids[i]) == outputs.end()) {
        std::vector<int> v{ir.results[i]};
        outputs[bc->token2ids.guids[i]] = v;
      } else {
        outputs[bc->token2ids.guids[i]].push_back(ir.results[i]);
      }
      /* std::cout << "outputs: ";
      for(const auto& elem : outputs){
        std::cout << elem.first << ": [";
        for (const auto &vel : elem.second) {
          std::cout << vel << " ";
        }
        std::cout << "]" << std::endl;
      } */
      // std::cout << "outputs[bc->token2ids.guids[i]].size(): " <<
      // outputs[bc->token2ids.guids[i]].size() << std::endl; std::cout << "i: "
      // << i << std::endl; std::cout <<
      // "bc->token2ids.token_indexes[i].token_position: " <<
      // bc->token2ids.token_indexes[i].token_position << std::endl; std::cout
      // << "bc->token2ids.token_indexes[i].initial_length: " <<
      // bc->token2ids.token_indexes[i].initial_length << std::endl;
      assert(outputs[bc->token2ids.guids[i]].size() ==
             (bc->token2ids.token_indexes[i].token_position + 1) -
                 (bc->token2ids.token_indexes[i].initial_length - 1));
      batch_predictions[bc->token2ids.guids[i]] = ir.results[i];
    }
  }
  assert(batch_predictions.size() == bc->num_active_requests());
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


#include "llama.h"
#include <random>

using namespace Legion;

DataLoader::DataLoader(FFModel &ff,
                       LLAMAConfig const *llamaconfig,
                       ParallelTensor const &input) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = llamaconfig->sentence_len;

  {
    batch_input = input;
    int num_dims = input->num_dims;

    ParallelDim dims[num_dims];
    for (int i = 0; i < num_dims; i++) {
      if (i == 0) {
        dims[i].size = 1;
      } else {
        dims[i].size = input->dims[i].size;
      }

      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = input->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == num_dims - 1 || (!dims[i].is_replica_dim));
    }
    // dims[num_dims - 1].size = num_samples;
    // full_input =
    //     ff.create_parallel_tensor_legion_ordering(num_dims, dims, DT_INT64);
    // assert(full_input != nullptr && "full_input is nullptr");
    // ff.map_tensor(full_input, NULL /*parallel_op*/);
  }

  // size_t llamaconfig_size = sizeof(llamaconfig);
  // std::cout << "llama config dataloader: " << llamaconfig->input_path;

  // // Load entire dataset
  // TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
  //                      TaskArgument(llamaconfig, llamaconfig_size));
  // // regions[1]: full_input
  // launcher.add_region_requirement(RegionRequirement(full_input->region,
  //                                                  WRITE_ONLY,
  //                                                  EXCLUSIVE,
  //                                                  full_input->region,
  //                                                  MAP_TO_FB_MEMORY));
  // launcher.add_field(0, FID_DATA);
  // runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  LLAMAConfig const *llamaconfig = (LLAMAConfig *)task->args;

  AccessorWO<long, 3> const acc_input(regions[0], FID_DATA);
  Rect<3> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));

  long *input_ptr = acc_input.ptr(rect_input.lo);
  std::cout << "load entire dataset" << rect_input.volume();

  // load from file
  load_from_file(input_ptr,
                 rect_input.volume(),
                 "/home/ubuntu/FlexFlow/examples/cpp/inference/LLAMA/tokens/"
                 "llama_demo_tokens");
}

void DataLoader::next_batch(FFModel &ff,
                            BatchConfig *bc,
                            std::map<size_t, long> &batch_predictions) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load Input
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_input->parallel_is);
    ArgumentMap argmap;

    DataLoaderNextBatchInput next_batch_input = {bc, batch_predictions};
    DataLoaderNextBatchInput const *ptr = &next_batch_input;
    size_t next_batch_input_sz = sizeof(next_batch_input);
    assert(ptr->prev_batch_preds.size() == batch_predictions.size());

    std::cout << "next batch internal" << std::endl;
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1,
                           batch_input->parallel_is,
                           TaskArgument(ptr, next_batch_input_sz),
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
  // progress next_index
  next_index += ff.config.batchSize;
  next_token_idx += 1;
}

void DataLoader::reset() {
  next_index = 0;
  next_token_idx = 0;
  next_batch_index = 0;
}

template <typename T>
void DataLoader::load_from_file(T *ptr, size_t size, std::string filename) {

  std::cout << "load from file: " << filename << std::endl;
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  std::vector<T> host_array(size);
  size_t loaded_data_size = sizeof(T) * size;
  in.seekg(0, in.end);
  in.seekg(0, in.beg);
  in.read((char *)host_array.data(), loaded_data_size);

  size_t in_get_size = in.gcount();
  // std::cout << "size seee" << std::endl;
  // std::cout << loaded_data_size << std::endl;
  // std::cout << in_get_size << std::endl;
  if (in_get_size != loaded_data_size) {
    std::cout << "load data error" << std::endl;
    return;
  }

  // std::cout << "finish loading input";
  assert(size == host_array.size());

  // normal
  long data_index = 0;
  for (auto v : host_array) {
    ptr[data_index++] = v;
  }
  in.close();
}

template <typename T>
void DataLoader::load_attention_weights(T *ptr,
                                        size_t size,
                                        std::string layer_name,
                                        std::string weight_path) {

  std::string q_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wq_weight";
  std::string k_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wk_weight";
  std::string v_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wv_weight";
  std::string o_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wo_weight";
  std::vector<std::string> weight_files = {q_file, k_file, v_file, o_file};

  size_t index = 0;
  int file_index = 0;

  // q, k, v, o -> 0, 1, 2, 3
  for (auto file : weight_files) {
    std::cout << "file name and index: " << file << "->" << file_index << "\n";
    size_t partial_size = size / 4;
    std::ifstream in(file, std::ios::in | std::ios::binary);
    std::vector<T> host_array(partial_size);
    size_t loaded_data_size = sizeof(T) * partial_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error";
      return;
    }
    assert(partial_size == host_array.size());

    size_t one_head_size = 4096 * 128;
    size_t data_index = 0;

    for (int i = 0; i < 32; i++) {
      size_t start_index = i * one_head_size * 4 + file_index * one_head_size;
      for (size_t j = start_index; j < start_index + one_head_size; j++) {
        ptr[j] = host_array.at(data_index);
        data_index += 1;
      }
    }
    file_index++;

    in.close();
    index++;
  }
}

void DataLoader::store_outputs(BatchConfig *bc,
                               InferenceResult const &ir,
                               std::map<size_t, long> &batch_predictions) {

  std::cout << "store outputs...." << std::endl;
  batch_predictions.clear();

  // size_t guid = bc->tokensInfo[0].guid;
  auto guid = bc->requestsInfo[bc->tokensInfo[0].request_index].request_guid;

  int start_idx = bc->tokensInfo[0].abs_depth_in_request;

  // only store the last token of each req
  for (size_t i = 0; i <= bc->num_active_tokens(); i++) {
    auto current_guid =
        bc->requestsInfo[bc->tokensInfo[i].request_index].request_guid;
    if (i == bc->num_active_tokens() || current_guid != guid) {

      int result_index = bc->tokensInfo[i - 1].abs_depth_in_request - start_idx;
      batch_predictions[guid] = ir.token_ids[i - 1];

      std::cout << "i: " << i << ", dds-" << guid << ", result index"
                << result_index << ", result value: " << batch_predictions[guid]
                << "\n";

      if (i < bc->num_active_tokens()) {
        guid = bc->requestsInfo[bc->tokensInfo[i].request_index].request_guid;
        start_idx = bc->tokensInfo[i].abs_depth_in_request;
      }
    }
  }

  assert(batch_predictions.size() == bc->num_active_requests());
}

template void DataLoader::load_attention_weights<float>(
    float *ptr, size_t size, std::string layer_name, std::string weight_path);
template void DataLoader::load_from_file<long>(long *ptr,
                                               size_t size,
                                               std::string filename);
template void DataLoader::load_from_file<float>(float *ptr,
                                                size_t size,
                                                std::string filename);

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
        registrar, "Load Inputs Task");
  }
}

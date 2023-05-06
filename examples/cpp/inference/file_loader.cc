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

#include "file_loader.h"
#include "flexflow/inference.h"

#include <vector>
using namespace std;

FileDataLoader::FileDataLoader(std::string _input_path,
                               std::string _weight_file_path)
    : input_path(_input_path), weight_file_path(_weight_file_path){};

BatchConfig::TokenId *FileDataLoader::generate_requests(int num, int length) {

  BatchConfig::TokenId *prompts =
      (BatchConfig::TokenId *)malloc(sizeof(BatchConfig::TokenId) * 40);
  std::cout << "load input from file: " << input_path << std::endl;
  std::ifstream in(input_path, std::ios::in | std::ios::binary);
  int size = num * length;
  std::vector<long> host_array(size);
  size_t loaded_data_size = sizeof(long) * size;

  std::cout << "loaded_data_size: " << loaded_data_size << std::endl;
  in.seekg(0, in.end);
  in.seekg(0, in.beg);
  in.read((char *)host_array.data(), loaded_data_size);

  std::cout << "loaded_data_size: " << loaded_data_size << std::endl;

  size_t in_get_size = in.gcount();
  if (in_get_size != loaded_data_size) {
    std::cout << "load data error";
    return prompts;
  }

  assert(size == host_array.size());

  int index = 0;
  int data_index = 0;

  std::cout << "loaded_data_size: " << loaded_data_size << std::endl;
  std::cout << host_array.size() << "\n";
  for (auto v : host_array) {
    prompts[data_index++] = v;
    std::cout << data_index << ", " << (int)v << "\n";
  }
  in.close();
  return prompts;
};

void load_attention_weights(float *ptr,
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
    std::vector<float> host_array(partial_size);
    size_t loaded_data_size = sizeof(float) * partial_size;
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

void load_from_file(float *ptr, size_t size, std::string filename) {
  std::cout << "load from file: " << filename << std::endl;
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  std::vector<float> host_array(size);
  size_t loaded_data_size = sizeof(float) * size;
  in.seekg(0, in.end);
  in.seekg(0, in.beg);
  in.read((char *)host_array.data(), loaded_data_size);

  size_t in_get_size = in.gcount();
  // std::cout << "size seee" << std::endl;
  // std::cout << loaded_data_size << std::endl;
  // std::cout << in_get_size << std::endl;
  if (in_get_size != loaded_data_size) {
    std::cout << "load data error";
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

void FileDataLoader::load_weights(
    FFModel *ff, std::unordered_map<std::string, Layer *> weights_layers) {

  for (auto &v : weights_layers) {
    Tensor weight = v.second->weights[0];
    std::cout << "weights layer: " << v.first << "\n";

    if (weight == NULL) {
      std::cout << "op no weights : " << v.first << "\n";
      continue;
    }

    size_t volume = 1;
    std::vector<int> dims_vec;
    for (int i = 0; i < weight->num_dims; i++) {
      dims_vec.push_back(weight->dims[i]);
      volume *= weight->dims[i];
    }

    assert(weight->data_type == DT_FLOAT);
    float *data = (float *)malloc(sizeof(float) * volume);

    if (v.first.find("attention_w") != std::string::npos) {
      load_attention_weights(data, volume, v.first, weight_file_path);

    } else {
      load_from_file(data, volume, weight_file_path + v.first);
    }

    ParallelTensor weight_pt;
    ff->get_parallel_tensor_from_tensor(weight, weight_pt);
    weight_pt->set_tensor<float>(ff, dims_vec, data);
  }
}

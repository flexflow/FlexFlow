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
                               std::string _weight_file_path,
                               int _num_heads,
                               size_t _hidden_dim,
                               size_t _qkv_inner_dim)
    : input_path(_input_path), weight_file_path(_weight_file_path),
      num_heads(_num_heads), hidden_dim(_hidden_dim),
      qkv_inner_dim(_qkv_inner_dim){};

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
    std::cout << "load data error" << std::endl;
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

void load_attention_weights(float *dst_ptr,
                            int num_heads,
                            size_t hidden_dim,
                            size_t qkv_inner_dim,
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

  int weight_index = 0; // {q, k, v, o} -> {0, 1, 2, 3}
  size_t single_proj_size =
      hidden_dim *
      qkv_inner_dim; // size of each of Q,K,V,O weights for a single head
  size_t one_head_size =
      single_proj_size * 4; // size of Q+K+V+O weights for a single head
  size_t one_weight_file_size =
      num_heads * single_proj_size; // size of each of Q/K/V/O for all heads

  for (auto file : weight_files) {
    std::cout << "file name and index: " << file << "->" << weight_index
              << "\n";
    std::ifstream in(file, std::ios::in | std::ios::binary);
    std::vector<float> host_array(one_weight_file_size);
    size_t loaded_data_size = sizeof(float) * one_weight_file_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error" << std::endl;
      return;
    }
    assert(one_weight_file_size == host_array.size());

    size_t flexflow_idx;
    for (int i = 0; i < one_weight_file_size; i++) {
      int checkpoint_row_idx = i % hidden_dim;
      int checkpoint_column_idx = (i / hidden_dim) % qkv_inner_dim;
      int head_idx = i / single_proj_size;
      if (weight_index < 3) {
        // if this is the Q,K or V weight
        flexflow_idx =
            head_idx * one_head_size + weight_index * single_proj_size +
            checkpoint_column_idx * hidden_dim + checkpoint_row_idx;
      } else {
        // if this is the output projection weight
        flexflow_idx =
            head_idx * one_head_size + weight_index * single_proj_size +
            checkpoint_row_idx * qkv_inner_dim + checkpoint_column_idx;
      }
      dst_ptr[flexflow_idx] = host_array.at(i);
    }

    weight_index++;

    in.close();
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
    std::cout << "load data error" << std::endl;
    return;
  }

  // std::cout << "finish loading input" << std::endl;
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
      assert(dims_vec[0] = hidden_dim * qkv_inner_dim * 4);
      assert(dims_vec[1] = num_heads);
      assert(volume == dims_vec[0] * dims_vec[1]);
      load_attention_weights(data,
                             num_heads,
                             hidden_dim,
                             qkv_inner_dim,
                             v.first,
                             weight_file_path);

    } else {
      load_from_file(data, volume, weight_file_path + v.first);
    }

    ParallelTensor weight_pt;
    ff->get_parallel_tensor_from_tensor(weight, weight_pt);
    weight_pt->set_tensor<float>(ff, dims_vec, data);
  }
}

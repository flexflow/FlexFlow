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
#include "flexflow/ffconst_utils.h"
#include "flexflow/inference.h"

#include <vector>
using namespace std;

using namespace Legion;

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
  std::ifstream in(input_path, std::ios::in | std::ios::binary);
  int size = num * length;
  std::vector<long> host_array(size);
  size_t loaded_data_size = sizeof(long) * size;

  in.seekg(0, in.end);
  in.seekg(0, in.beg);
  in.read((char *)host_array.data(), loaded_data_size);

  size_t in_get_size = in.gcount();
  if (in_get_size != loaded_data_size) {
    std::cout << "load data error" << std::endl;
    return prompts;
  }

  assert(size == host_array.size());
  int index = 0;
  int data_index = 0;

  for (auto v : host_array) {
    prompts[data_index++] = v;
  }
  in.close();
  return prompts;
};

template <typename DT>
void load_attention_bias(DT *ptr,
                         int num_heads,
                         size_t hidden_dim,
                         size_t qkv_inner_dim,
                         std::string layer_name,
                         std::string weight_path) {
  std::string q_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wq_bias";
  std::string k_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wk_bias";
  std::string v_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wv_bias";
  std::string o_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_wo_bias";
  std::vector<std::string> bias_files = {q_file, k_file, v_file, o_file};

  int file_index = 0;
  for (auto file : bias_files) {
    size_t partial_size = hidden_dim;
    // std::cout << "Loading filename: " << file << std::endl;
    std::ifstream in(file, std::ios::in | std::ios::binary);
    assert(in.good() && "incorrect bias file path");
    std::vector<DT> host_array(partial_size);
    size_t loaded_data_size = sizeof(DT) * partial_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load bias data error";
      return;
    }
    assert(partial_size == host_array.size());

    size_t data_index = 0;

    for (int i = 0; i < hidden_dim; i++) {
      ptr[file_index * hidden_dim + i] = host_array.at(data_index);
      data_index++;
    }

    file_index++;

    in.close();
  }
}

template <typename DT>
void load_attention_weights_multi_query(DT *ptr,
                                        std::string layer_name,
                                        std::string weight_path,
                                        size_t hidden_dim,
                                        int num_heads) {

  std::string qkv_file = weight_path +
                         layer_name.substr(0, layer_name.find("attention")) +
                         "attention_query_key_value_weight";
  std::string o_file = weight_path +
                       layer_name.substr(0, layer_name.find("attention")) +
                       "attention_dense_weight";

  // q has n_heads heads, k and v only have one head, o have n_head heads
  std::vector<std::string> weight_files = {qkv_file, o_file};
  int file_index = 0;
  int data_index = 0;
  for (auto file : weight_files) {
    size_t partial_size =
        file_index == 0 ? (hidden_dim + 2 * hidden_dim / num_heads) * hidden_dim
                        : hidden_dim * hidden_dim;

    std::ifstream in(file, std::ios::in | std::ios::binary);
    // std::cout << "Loading filename: " << file << std::endl;
    if (!in.good()) {
      std::cout << "Could not open file: " << file << std::endl;
    }
    assert(in.good() && "incorrect weight file path");
    std::vector<DT> host_array(partial_size);
    size_t loaded_data_size = sizeof(DT) * partial_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error " << in_get_size << ", "
                << loaded_data_size;
      assert(false && "data size mismatch");
    }
    for (int i = 0; i < partial_size; i++) {
      ptr[data_index++] = host_array.at(i);
    }
    file_index++;
  }
}

template <typename DT>
void load_attention_weights(DT *ptr,
                            int num_heads,
                            size_t hidden_dim,
                            size_t qkv_inner_dim,
                            std::string layer_name,
                            std::string weight_path,
                            size_t volume) {
  // layers_0_attention_wq_weight
  // layers_0_self_attn_q_proj_weight
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

  int file_index = 0;

  size_t single_proj_size =
      hidden_dim *
      qkv_inner_dim; // size of each of Q,K,V,O weights for a single head
  size_t one_weight_file_size =
      num_heads * single_proj_size; // size of each of Q/K/V/O for all heads

  // q, k, v, o -> 0, 1, 2, 3
  for (auto file : weight_files) {
    size_t partial_size = one_weight_file_size;

    std::ifstream in(file, std::ios::in | std::ios::binary);
    // std::cout << "Loading filename: " << file << std::endl;
    if (!in.good()) {
      std::cout << "Could not open file: " << file << std::endl;
    }
    assert(in.good() && "incorrect weight file path");
    std::vector<DT> host_array(partial_size);
    size_t loaded_data_size = sizeof(DT) * partial_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error";
      return;
    }
    assert(partial_size == host_array.size());

    size_t one_head_size = hidden_dim * (hidden_dim / num_heads);
    size_t data_index = 0;

    for (int i = 0; i < num_heads; i++) {
      size_t start_index = i * one_head_size * 4 + file_index * one_head_size;
      for (size_t j = start_index; j < start_index + one_head_size; j++) {
        ptr[j] = host_array.at(data_index);
        data_index += 1;
      }
    }
    file_index++;

    in.close();
  }
}

template <typename DT>
void load_from_file(DT *ptr, size_t size, std::string filename) {
  // std::cout << "Loading filename: " << filename << std::endl;
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (!in.good()) {
    std::cout << "Could not open file: " << filename << std::endl;
  }
  assert(in.good() && "incorrect weight file path");
  std::vector<DT> host_array(size);
  size_t loaded_data_size = sizeof(DT) * size;
  in.seekg(0, in.end);
  in.seekg(0, in.beg);
  in.read((char *)host_array.data(), loaded_data_size);

  size_t in_get_size = in.gcount();
  if (in_get_size != loaded_data_size) {
    std::cout << "load weight data error " << in_get_size << ", "
              << loaded_data_size << ", " << sizeof(DT) << std::endl;
    return;
  }
  assert(size == host_array.size());

  // normal
  long data_index = 0;
  for (auto v : host_array) {
    ptr[data_index++] = v;
  }
  in.close();
}

void FileDataLoader::load_positions(FFModel *ff,
                                    Tensor pt,
                                    ParallelTensor position_pt,
                                    int max_seq_length,
                                    int offset) {
  size_t volume = 1;
  std::vector<int> dims_vec;
  for (int i = 0; i < pt->num_dims; i++) {
    volume *= pt->dims[i];
    dims_vec.push_back(pt->dims[i]);
  }

  // load data;
  int *data = (int *)malloc(sizeof(int) * volume);
  for (int i = 0; i < volume; i++) {
    data[i] = i % max_seq_length + offset;
  }
  // set tensor

  // ParallelTensor position_pt;

  // ff->get_parallel_tensor_from_tensor(pt, position_pt);
  position_pt->set_tensor<int>(ff, dims_vec, data);
}

template <typename DT>
void FileDataLoader::load_single_weight_tensor(FFModel *ff,
                                               Tensor weight,
                                               int weight_idx,
                                               std::string const &layername) {
  size_t volume = 1;
  std::vector<int> dims_vec;
  for (int i = 0; i < weight->num_dims; i++) {
    dims_vec.push_back(weight->dims[i]);
    volume *= weight->dims[i];
  }

  assert(data_type_size(weight->data_type) == sizeof(DT));
  DT *data = (DT *)malloc(sizeof(DT) * volume);

  std::string file_path =
      (layername.back() == '/') ? layername : "/" + layername;

  if (file_path.find("attention_w") != std::string::npos) {
    if (weight_idx == 0) {
      load_attention_weights(data,
                             num_heads,
                             hidden_dim,
                             qkv_inner_dim,
                             file_path,
                             weight_file_path,
                             volume);
    } else {
      load_attention_bias(data,
                          num_heads,
                          hidden_dim,
                          qkv_inner_dim,
                          file_path,
                          weight_file_path);
    }

  } else if (file_path.find("self_attention") != std::string::npos) {
    load_attention_weights_multi_query(
        data, file_path, weight_file_path, hidden_dim, num_heads);
  } else {
    if (weight_idx > 0) {
      int index = file_path.find("_weight");
      assert(index != std::string::npos);
      file_path = file_path.substr(0, index) + "_bias";
    }
    load_from_file(data, volume, weight_file_path + file_path);
  }

  ParallelTensor weight_pt;
  ff->get_parallel_tensor_from_tensor(weight, weight_pt);
  weight_pt->set_tensor<DT>(ff, dims_vec, data);

  delete data;
}

void FileDataLoader::load_weights(
    FFModel *ff, std::unordered_map<std::string, Layer *> weights_layers) {
  for (auto &v : weights_layers) {
    int weights_num = v.second->numWeights;
    for (int i = 0; i < weights_num; i++) {
      Tensor weight = v.second->weights[i];
      if (weight == NULL) {
        continue;
      }
      switch (weight->data_type) {
        case DT_HALF:
          load_single_weight_tensor<half>(ff, weight, i, v.first);
          break;
        case DT_FLOAT:
          load_single_weight_tensor<float>(ff, weight, i, v.first);
          break;
        default:
          assert(false && "Unsupported data type");
      }
    }
  }
}

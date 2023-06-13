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
                               int _tensor_parallelism_degree,
                               int _num_heads,
                               size_t _hidden_dim,
                               size_t _qkv_inner_dim)
    : input_path(_input_path), weight_file_path(_weight_file_path),
      tensor_parallelism_degree(_tensor_parallelism_degree),
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
                         int partition_idx,
                         int tensor_parallelism_degree,
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

  assert(num_heads % tensor_parallelism_degree == 0);

  int file_index = 0;

  for (auto file : bias_files) {
    size_t qkv_partial_size =
        qkv_inner_dim * (num_heads / tensor_parallelism_degree);
    size_t out_partial_size = hidden_dim;
    size_t partial_size =
        (file_index < 3) ? qkv_partial_size : out_partial_size;
    // std::cout << "Loading attention bias filename: " << file << std::endl;
    // std::cout << "partition_idx: " << partition_idx
    //   << ", tensor_parallelism_degree: " << tensor_parallelism_degree
    //   << ", num_heads: " << num_heads
    //   << ", hidden_dim: " << hidden_dim
    //   << ", qkv_inner_dim: " << qkv_inner_dim
    //   << ", partial_size: " << partial_size << std::endl;
    std::ifstream in(file, std::ios::in | std::ios::binary);
    assert(in.good() && "incorrect bias file path");
    std::vector<DT> host_array(partial_size);
    size_t loaded_data_size = sizeof(DT) * partial_size;
    in.seekg(0, in.end);
    if (file_index < 3) {
      in.seekg(loaded_data_size * partition_idx, in.beg);
    } else {
      in.seekg(0, in.beg);
    }
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      printf(
          "load bias data error: in_get_size (%lu) != loaded_data_size (%lu)\n",
          in_get_size,
          loaded_data_size);
      assert(false);
    }
    assert(partial_size == host_array.size());

    size_t data_index = 0;

    for (int i = 0; i < partial_size; i++) {
      ptr[file_index * qkv_partial_size + i] = host_array.at(data_index);
      data_index++;
    }

    file_index++;

    in.close();
  }
}

template <typename DT>
void load_attention_weights(DT *ptr,
                            int partition_idx,
                            int tensor_parallelism_degree,
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
  std::vector<std::string> weight_files = {q_file, k_file, v_file};

  int file_index = 0;

  assert(num_heads % tensor_parallelism_degree == 0);

  size_t single_proj_size =
      hidden_dim *
      qkv_inner_dim; // size of each of Q,K,V,O weights for a single head
  size_t one_weight_file_size =
      (num_heads / tensor_parallelism_degree) *
      single_proj_size; // size of each of Q/K/V/O for all heads

  // q, k, v -> 0, 1, 2
  for (auto file : weight_files) {
    size_t partial_size = one_weight_file_size;

    std::ifstream in(file, std::ios::in | std::ios::binary);
    std::cout << "Loading attention filename: " << file << std::endl;
    if (!in.good()) {
      std::cout << "Could not open file: " << file << std::endl;
    }
    assert(in.good() && "incorrect weight file path");
    std::vector<DT> host_array(partial_size);
    size_t loaded_data_size = sizeof(DT) * partial_size;
    in.seekg(0, in.end);
    in.seekg(single_proj_size * partition_idx *
                 (num_heads / tensor_parallelism_degree) * sizeof(DT),
             in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error" << std::endl;
      assert(false);
    }
    assert(partial_size == host_array.size());

    size_t data_index = 0;
    for (int i = 0; i < num_heads / tensor_parallelism_degree; i++) {
      size_t start_index =
          i * single_proj_size * 4 + file_index * single_proj_size;
      for (size_t j = start_index; j < start_index + single_proj_size; j++) {
        ptr[j] = host_array.at(data_index);
        data_index += 1;
      }
    }
    assert(data_index == partial_size);
    file_index++;

    in.close();
  }

  // output weight file gets special treatment
  {
    std::ifstream in(o_file, std::ios::in | std::ios::binary);
    std::cout << "Loading attention filename: " << o_file << std::endl;
    if (!in.good()) {
      std::cout << "Could not open file: " << o_file << std::endl;
    }
    assert(in.good() && "incorrect weight file path");
    size_t full_output_weight_size = num_heads * single_proj_size;
    std::vector<DT> host_array(full_output_weight_size);
    size_t loaded_data_size = sizeof(DT) * full_output_weight_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error" << std::endl;
      assert(false);
    }
    assert(full_output_weight_size == host_array.size());

    for (int i = 0; i < num_heads / tensor_parallelism_degree; i++) {
      size_t start_index = i * single_proj_size * 4 + 3 * single_proj_size;
      for (size_t j = 0; j < single_proj_size; j++) {
        int ff_row_idx = j % hidden_dim;
        int ff_col_idx = j / hidden_dim;
        assert(ff_row_idx < hidden_dim && ff_col_idx < qkv_inner_dim);
        int abs_head_idx =
            i + partition_idx * (num_heads / tensor_parallelism_degree);
        size_t data_index = ff_row_idx * (qkv_inner_dim * num_heads) +
                            qkv_inner_dim * abs_head_idx + ff_col_idx;
        ptr[j + start_index] = host_array.at(data_index);
      }
    }

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
    assert(false);
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
    bool ends_with_slash = file_path.back() == '/';
    size_t underscorePosition = file_path.find_last_of("_");
    std::string numberSubstring = file_path.substr(underscorePosition + 1);
    std::string file_path2 = file_path.substr(0, underscorePosition);
    if (ends_with_slash) {
      numberSubstring.pop_back();
      file_path2 += '/';
    }
    int partition_idx = std::stoi(numberSubstring);
    assert(partition_idx >= 0 && partition_idx < tensor_parallelism_degree);
    // std::cout << "Loading file_path: " << file_path
    //           << ", file_path2: " << file_path2
    //           << ", partition_idx: " << partition_idx
    //           << ", weight_idx: " << weight_idx << std::endl;

    // std::cout << "data array has volume " << volume << std::endl;

    if (weight_idx == 0) {
      load_attention_weights(data,
                             partition_idx,
                             tensor_parallelism_degree,
                             num_heads,
                             hidden_dim,
                             qkv_inner_dim,
                             file_path2,
                             weight_file_path,
                             volume);
    } else {
      load_attention_bias(data,
                          partition_idx,
                          tensor_parallelism_degree,
                          num_heads,
                          hidden_dim,
                          qkv_inner_dim,
                          file_path2,
                          weight_file_path);
    }

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

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
                               int _num_kv_heads,
                               size_t _hidden_dim,
                               size_t _qkv_inner_dim,
                               int _tensor_partition_num)
    : input_path(_input_path), weight_file_path(_weight_file_path),
      num_heads(_num_heads), num_kv_heads(_num_kv_heads),
      hidden_dim(_hidden_dim), qkv_inner_dim(_qkv_inner_dim),
      tensor_partition_num(_tensor_partition_num){};

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
void load_attention_bias_v2(DT *ptr,
                            int num_heads,
                            int num_kv_heads,
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

  // now only opt use this.
  assert(num_heads == num_kv_heads);

  for (auto file : bias_files) {
    int n_heads = file_index == 0 ? num_heads : num_kv_heads;
    size_t qkv_partial_size = qkv_inner_dim * n_heads;
    size_t out_partial_size = hidden_dim;
    size_t partial_size =
        (file_index < 3) ? qkv_partial_size : out_partial_size;
    std::ifstream in(file, std::ios::in | std::ios::binary);
    assert(in.good() && "incorrect bias file path");
    std::vector<DT> host_array(partial_size);
    size_t loaded_data_size = sizeof(DT) * partial_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
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
void load_attention_weights_v2(DT *ptr,
                               int num_heads,
                               int num_kv_heads,
                               size_t hidden_dim,
                               size_t qkv_inner_dim,
                               std::string layer_name,
                               std::string weight_path,
                               size_t volume,
                               int tensor_partition_num) {
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

  int base_index = 0;
  size_t single_proj_size =
      hidden_dim *
      qkv_inner_dim; // size of each of Q,K,V,O weights for a single head
  size_t one_weight_file_size =
      num_heads * single_proj_size; // size of each of Q/K/V/O for all heads

  size_t q_size = one_weight_file_size, o_size = one_weight_file_size;
  size_t k_size = single_proj_size * num_kv_heads,
         v_size = single_proj_size * num_kv_heads;

  // stride for q, k, v, o
  size_t stride_size =
      (q_size + v_size + k_size + o_size) / tensor_partition_num;
  for (auto file : weight_files) {
    int data_index = 0;
    size_t partial_size = (file_index == 0 || file_index == 3)
                              ? one_weight_file_size
                              : single_proj_size * num_kv_heads;
    size_t one_partition_size = partial_size / tensor_partition_num;

    std::ifstream in(file, std::ios::in | std::ios::binary);
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
      std::cout << "load attention data error " << in_get_size << ", "
                << loaded_data_size;
      assert(false && "data size mismatch");
    }
    // wq, wk, wo
    for (int i = 0; i < tensor_partition_num; i++) {
      for (int j = 0; j < one_partition_size; j++) {
        ptr[base_index + i * stride_size + j] = host_array.at(data_index++);
      }
    }
    assert(data_index == partial_size);
    base_index += one_partition_size;
    file_index++;
  }
  assert(base_index == (q_size + k_size + v_size) / tensor_partition_num);

  {
    std::ifstream in(o_file, std::ios::in | std::ios::binary);
    if (!in.good()) {
      std::cout << "Could not open file: " << o_file << std::endl;
    }
    assert(in.good() && "incorrect weight file path");
    std::vector<DT> host_array(one_weight_file_size);
    size_t loaded_data_size = sizeof(DT) * one_weight_file_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error" << std::endl;
      assert(false);
    }
    assert(one_weight_file_size == host_array.size());
    int data_index = 0;

    int one_partition_size = qkv_inner_dim * (num_heads / tensor_partition_num);
    for (int i = 0; i < one_weight_file_size; i++) {
      int part_idx = (i / one_partition_size) % tensor_partition_num;
      int block_num = (i / one_partition_size);
      int offset = block_num / tensor_partition_num * one_partition_size +
                   (i % one_partition_size);
      ptr[base_index + part_idx * stride_size + offset] =
          host_array.at(data_index++);
    }

    in.close();

    assert(data_index == one_weight_file_size);
  }
}

template <typename DT>
void load_from_file(DT *ptr, size_t size, std::string filename) {
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

//--------------------- quantization functions ----------------------
// the data layout is 32 * quantized data + 1 scaling factor + 1 offset factor
// in the decompression mode, the real data = quantized data * scaling factor +
// offset

void load_attention_weights_quantized(char *ptr,
                                      int num_heads,
                                      size_t hidden_dim,
                                      size_t qkv_inner_dim,
                                      std::string layer_name,
                                      std::string weight_path,
                                      DataType data_type,
                                      bool use_full_precision) {
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
    if (!in.good()) {
      std::cout << "Could not open file: " << file << std::endl;
    }
    assert(in.good() && "incorrect weight file path");
    std::vector<char> host_array(partial_size);
    size_t loaded_data_size = sizeof(char) * partial_size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
    in.read((char *)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();

    if (in_get_size != loaded_data_size) {
      std::cout << "load data error";
      return;
    }
    assert(partial_size == host_array.size());

    size_t one_head_size = data_type == DT_INT8
                               ? hidden_dim * (hidden_dim / num_heads)
                               : hidden_dim * (hidden_dim / num_heads) / 2;

    size_t data_index = 0;
    for (int i = 0; i < num_heads; i++) {
      size_t start_index = i * one_head_size * 4 + file_index * one_head_size;
      for (size_t j = start_index; j < start_index + one_head_size; j++) {
        if (data_type == DT_INT4) {
          char v1 = host_array.at(data_index);
          char v2 = host_array.at(data_index + 1);
          ptr[j] = (v2 & 0XF) | (v1 << 4);
          data_index += 2;
        } else {
          ptr[j] = host_array.at(data_index);
          data_index += 1;
        }
      }
    }
    file_index++;
    in.close();
  }

  // load scale and offset to the end of weight tensor
  // the layout is like |values * 32 heads|offset|scale|
  size_t offset = data_type == DT_INT8 ? one_weight_file_size * 4
                                       : (one_weight_file_size * 4) / 2;
  for (auto file : weight_files) {
    for (int i = 0; i < 2; i++) {
      std::string meta_file = i == 0 ? (file + "_offset") : (file + "_scale");
      size_t partial_size =
          one_weight_file_size / INT4_NUM_OF_ELEMENTS_PER_GROUP;
      std::ifstream in(meta_file, std::ios::in | std::ios::binary);
      if (!in.good()) {
        std::cout << "Could not open file: " << meta_file << std::endl;
      }
      assert(in.good() && "incorrect weight file path");

      if (use_full_precision) {
        // float
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

        for (auto v : host_array) {
          *(float *)(ptr + offset) = v;
          offset += sizeof(float);
        }
      } else {
        // half
        std::vector<half> host_array(partial_size);
        size_t loaded_data_size = sizeof(half) * partial_size;
        in.seekg(0, in.end);
        in.seekg(0, in.beg);
        in.read((char *)host_array.data(), loaded_data_size);
        size_t in_get_size = in.gcount();

        if (in_get_size != loaded_data_size) {
          std::cout << "load data error";
          return;
        }
        assert(partial_size == host_array.size());
        for (auto v : host_array) {
          *(half *)(ptr + offset) = v;
          offset += sizeof(half);
        }
      }
    }
  }
}

void load_from_quantized_file(char *ptr,
                              size_t size,
                              std::string filename,
                              DataType data_type,
                              bool use_full_precision) {
  assert(data_type == DT_INT4 || data_type == DT_INT8);

  std::string value_file = filename;
  std::string offset_file = filename + "_offset";
  std::string scaling_file = filename + "_scale";
  size_t value_size = 0, offset_size = 0, scaling_size = 0;

  if (data_type == DT_INT4) {
    // float/half + 4bit quantization
    // size1 = volume / 2, size2 = volume / 32 * (sizeof(DT)), size3 = size2
    value_size = 2 * (use_full_precision ? (size * 2 / 3) : (size * 4 / 5));
    offset_size = use_full_precision ? (size / 6) : (size / 10);
    scaling_size = use_full_precision ? (size / 6) : (size / 10);
  } else if (data_type == DT_INT8) {
    // float/half + 8bit quantization
    // size1 = volume * 1, size2 = volume / 32 * (sizeof(DT)), size3 = size2
    value_size = use_full_precision ? (size * 4 / 5) : (size * 8 / 9);
    offset_size = use_full_precision ? (size / 10) : (size / 18);
    scaling_size = use_full_precision ? (size / 10) : (size / 18);
  }

  std::vector<std::string> quantized_files = {
      value_file, offset_file, scaling_file};
  std::vector<size_t> quantized_sizes = {value_size, offset_size, scaling_size};

  int file_idx = 0;
  long data_index = 0;
  for (auto file : quantized_files) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.good()) {
      std::cout << "Could not open file: " << file << std::endl;
    }
    assert(in.good() && "incorrect weight file path");

    // value file, every element is in one byte
    if (file_idx == 0) {
      size = quantized_sizes.at(file_idx);
      std::vector<char> host_array(size);
      size_t loaded_data_size = size;
      in.seekg(0, in.end);
      in.seekg(0, in.beg);
      in.read((char *)host_array.data(), loaded_data_size);

      size_t in_get_size = in.gcount();
      if (in_get_size != loaded_data_size) {
        std::cout << "load weight data error quantized" << in_get_size << ", "
                  << loaded_data_size << ", " << sizeof(char) << std::endl;
        return;
      }
      assert(size == host_array.size());

      // normal
      size_t idx = 0;
      while (idx < host_array.size()) {
        if (data_type == DT_INT4) {
          // pack 2 elements into one byte
          char v1 = host_array.at(idx);
          char v2 = host_array.at(idx + 1);
          // v1 in first 4 bit and v2 in last 4 bit;
          ptr[data_index++] = (v2 & 0XF) | (v1 << 4);
          idx += 2;
        } else {
          ptr[data_index++] = host_array.at(idx++);
        }
      }
    } else if (use_full_precision) {
      // load offset/scale in float type;
      size = quantized_sizes.at(file_idx);
      std::vector<float> host_array(size / sizeof(float));
      size_t loaded_data_size = size;
      in.seekg(0, in.end);
      in.seekg(0, in.beg);
      in.read((char *)host_array.data(), loaded_data_size);

      size_t in_get_size = in.gcount();
      if (in_get_size != loaded_data_size) {
        std::cout << "load weight data error scale/offset" << in_get_size
                  << ", " << loaded_data_size << ", " << sizeof(float) << ", "
                  << file << ", " << size << std::endl;
        return;
      }
      assert(size / sizeof(float) == host_array.size());
      for (auto v : host_array) {
        *(float *)(ptr + data_index) = v;
        data_index += sizeof(float);
      }

    } else {
      // load offset/scale in half type;
      size = quantized_sizes.at(file_idx);
      std::vector<half> host_array(size / sizeof(half));
      size_t loaded_data_size = size;
      in.seekg(0, in.end);
      in.seekg(0, in.beg);
      in.read((char *)host_array.data(), loaded_data_size);

      size_t in_get_size = in.gcount();
      if (in_get_size != loaded_data_size) {
        std::cout << "load weight data error " << in_get_size << ", "
                  << loaded_data_size << ", " << sizeof(half) << std::endl;
        return;
      }
      assert(size / sizeof(half) == host_array.size());
      // normal
      for (auto v : host_array) {
        *(half *)(ptr + data_index) = v;
        data_index += sizeof(half);
      }
    }
    in.close();
    file_idx++;
  }
}

void FileDataLoader::load_quantization_weight(FFModel *ff,
                                              Tensor weight,
                                              int weight_idx,
                                              std::string const &layername,
                                              bool use_full_precision) {
  size_t volume = 1;
  std::vector<int> dims_vec;
  for (int i = 0; i < weight->num_dims; i++) {
    dims_vec.push_back(weight->dims[i]);
    volume *= weight->dims[i];
  }

  char *data = (char *)malloc(sizeof(char) * volume);

  std::string file_path =
      (layername.back() == '/') ? layername : "/" + layername;

  if (file_path.find("attention_w") != std::string::npos) {
    if (weight_idx == 0) {
      load_attention_weights_quantized(data,
                                       num_heads,
                                       hidden_dim,
                                       qkv_inner_dim,
                                       file_path,
                                       weight_file_path,
                                       weight->data_type,
                                       use_full_precision);
    }
    // else {
    //   load_attention_bias_quantized(data,
    //                                 num_heads,
    //                                 hidden_dim,
    //                                 qkv_inner_dim,
    //                                 file_path,
    //                                 weight_file_path);
    // }

  } else {
    if (weight_idx > 0) {
      int index = file_path.find("_weight");
      assert(index != std::string::npos);
      file_path = file_path.substr(0, index) + "_bias";
    }
    load_from_quantized_file(data,
                             volume,
                             weight_file_path + file_path,
                             weight->data_type,
                             use_full_precision);
  }

  ParallelTensor weight_pt;
  ff->get_parallel_tensor_from_tensor(weight, weight_pt);
  weight_pt->set_tensor<char>(ff, dims_vec, data);

  delete data;
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

  std::cout << "load weights: " << layername << "\n";

  assert(data_type_size(weight->data_type) == sizeof(DT));
  DT *data = (DT *)malloc(sizeof(DT) * volume);

  std::string file_path =
      (layername.back() == '/') ? layername : "/" + layername;

  if (file_path.find("attention_w") != std::string::npos) {
    if (weight_idx == 0) {
      load_attention_weights_v2(data,
                                num_heads,
                                num_kv_heads,
                                hidden_dim,
                                qkv_inner_dim,
                                file_path,
                                weight_file_path,
                                volume,
                                tensor_partition_num);
    } else {
      load_attention_bias_v2(data,
                             num_heads,
                             num_kv_heads,
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
    FFModel *ff,
    std::unordered_map<std::string, Layer *> weights_layers,
    bool use_full_precision) {
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
        case DT_INT4:
        case DT_INT8:
          // load weights in quantization
          load_quantization_weight(ff, weight, i, v.first, use_full_precision);
          break;
        default:
          assert(false && "Unsupported data type");
      }
    }
  }
}

/* Copyright 2019 Stanford
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

#include "config.h"
#include "simulator.h"
#include <fstream>
#include <iostream>
#include <string>

MappingTagID FFConfig::get_hash_id(const std::string& pcname)
{
  return std::hash<std::string>{}(pcname);
}

bool FFConfig::find_parallel_config(int ndims,
                                    const std::string& pcname,
                                    ParallelConfig& config) const
{
  MappingTagID hash = get_hash_id(pcname);
  std::map<MappingTagID, ParallelConfig>::const_iterator iter;
  if (strategies.find(hash) == strategies.end()) {
    // No strategy found, use default data parallelism
    switch (ndims) {
      case 1:
      {
        iter = strategies.find(DataParallelism_GPU_1D);
        assert(iter != strategies.end());
        config = iter->second;
        break;
      }
      case 2:
      {
        iter = strategies.find(DataParallelism_GPU_2D);
        assert(iter != strategies.end());
        config = iter->second;
        break;
      }
      case 3:
      {
        iter = strategies.find(DataParallelism_GPU_3D);
        assert(iter != strategies.end());
        config = iter->second;
        break;
      }
      case 4:
      {
        iter = strategies.find(DataParallelism_GPU_4D);
        assert(iter != strategies.end());
        config = iter->second;
        break;
      }
#if MAX_TENSOR_DIM >= 5
      case 5:
      {
        iter = strategies.find(DataParallelism_GPU_5D);
        assert(iter != strategies.end());
        config = iter->second;
        break;
      }
#endif
      //case 6:
      //{
      //  assert(strategies.find(DataParallelism_GPU_6D) != strategies.end());
      //  config = strategies[DataParallelism_GPU_6D];
      //  break;
      //}
      default:
      {
        // Unsupported dimension for data parallelism
        assert(false);
      }
    }
    return true;
  } else {
    iter = strategies.find(hash);
    config = iter->second;
    // Check that the returned config matches what we are looking for
    assert(config.nDims == ndims);
    return true;
  }
}

bool load_strategies_from_file(const std::string& filename,
                               std::map<MappingTagID, ParallelConfig>& strategies)
{
  std::fstream input(filename, std::ios::in);
  if (!input) {
    std::cerr << "Failed to open strategy file for reading" << std::endl;
    return false;
  }

  int ops_size = 0;
  input >> ops_size; 
  for (int i = 0; i < ops_size; i++) {
    ParallelConfig config;
    char op_name[MAX_OPNAME];
    int device_type_;
    input >> op_name;
    input >> device_type_;
    //printf("%s, %d\n", op_name, device_type_);
    ParallelConfig::DeviceType device_type = static_cast<ParallelConfig::DeviceType>(device_type_);
    switch (device_type) {
      case ParallelConfig::GPU:
      case ParallelConfig::CPU:
        config.device_type = device_type;
        break;
      default:
        fprintf(stderr, "Unsupported Device Type\n");
        assert(false);
    }
    input >> config.nDims;
    //printf("ndims %d\n", config.nDims);
    int n = 1;
    for (int j = 0; j < config.nDims; j++) {
      input >> config.dim[j];
      n = n * config.dim[j];
      //printf("%d\t", config.dim[j]);
    }
    //printf("\n");
    int device_ids_size = 0;
    input >> device_ids_size;
    //printf("device size %d\n", device_ids_size);
    assert(n == device_ids_size || device_ids_size == 0);
    for (int j = 0; j < device_ids_size; j++) {
      input >> config.device_ids[j];
      //printf("%d\t", config.device_ids[j]);
    }
    //printf("\n");
    MappingTagID hash = FFConfig::get_hash_id(op_name);
    assert(strategies.find(hash) == strategies.end());
    strategies[hash] = config;
  }
  input.close();
  printf("strategies.size() = %zu\n", strategies.size());
  return true;
}

bool save_strategies_to_file(const std::string& filename,
                             const std::map<std::string, ParallelConfig>& strategies)
{
  std::fstream output(filename, std::ios::out | std::ios::trunc);
  if (!output) {
    std::cerr << "Failed to open strategy file for writing!" << std::endl;
    return false;
  }
  
  output << strategies.size() << std::endl;   
  std::map<std::string, ParallelConfig>::const_iterator it;
  for (it = strategies.begin(); it != strategies.end(); it++) {
    output << it->first << std::endl;
    ParallelConfig config = it->second;
    switch (config.device_type) {
      case ParallelConfig::GPU:
      case ParallelConfig::CPU:
        output << config.device_type << std::endl;
        break;
      default:
        fprintf(stderr, "Unsupported Device Type\n");
        assert(false);
    }
    int n = 1;
    output << config.nDims << std::endl;
    for (int j = 0; j < config.nDims; j++) {
      n = n * config.dim[j];
      output << config.dim[j] << '\t';
    }
    output << std::endl;
    output << n << std::endl;
    for (int j = 0; j < n; j++) {
      output << config.device_ids[j] << '\t';
    }
    output << std::endl;
  }
  
  output.close();
  return true;
}

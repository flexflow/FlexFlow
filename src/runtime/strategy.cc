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

#include "strategy.pb.h"
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
  FFProtoBuf::Strategy strategyPb;
  std::fstream input(filename, std::ios::in);
  if (!strategyPb.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse strategy file" << std::endl;
    return false;
  }

  for (int i = 0; i < strategyPb.ops_size(); i++) {
    const FFProtoBuf::Op& op = strategyPb.ops(i);
    ParallelConfig config;
    switch (op.device_type()) {
      case FFProtoBuf::Op_DeviceType_GPU:
        config.device_type = ParallelConfig::GPU;
        break;
      case FFProtoBuf::Op_DeviceType_CPU:
        config.device_type = ParallelConfig::CPU;
        break;
      default:
        fprintf(stderr, "Unsupported Device Type\n");
        assert(false);
    }
    config.nDims = op.dims_size();
    int n = 1;
    for (int j = 0; j < config.nDims; j++) {
      config.dim[j] = op.dims(j);
      n = n * config.dim[j];
    }
    assert(n == op.device_ids_size() || op.device_ids_size() == 0);
    for (int j = 0; j < op.device_ids_size(); j++)
      config.device_ids[j] = op.device_ids(j);
    MappingTagID hash = FFConfig::get_hash_id(op.name());
    assert(strategies.find(hash) == strategies.end());
    strategies[hash] = config;
  }
  printf("strategies.size() = %zu\n", strategies.size());
  return true;
}

bool save_strategies_to_file(const std::string& filename,
                             const std::map<std::string, ParallelConfig>& strategies)
{
  FFProtoBuf::Strategy strategyPb;
  std::map<std::string, ParallelConfig>::const_iterator it;
  for (it = strategies.begin(); it != strategies.end(); it++) {
    FFProtoBuf::Op* op = strategyPb.add_ops();
    ParallelConfig config = it->second;
    switch (config.device_type) {
      case ParallelConfig::GPU:
        op->set_device_type(FFProtoBuf::Op_DeviceType_GPU);
        break;
      case ParallelConfig::CPU:
        op->set_device_type(FFProtoBuf::Op_DeviceType_CPU);
        break;
      default:
        fprintf(stderr, "Unsupported Device Type\n");
        assert(false);
    }
    op->set_name(it->first);
    int n = 1;
    for (int j = 0; j < config.nDims; j++) {
      n = n * config.dim[j];
      op->add_dims(config.dim[j]);
    }
    for (int j = 0; j < n; j++) {
      op->add_device_ids(config.device_ids[j]);
    }
  }
  std::fstream output(filename, std::ios::out | std::ios::trunc);
  if (!strategyPb.SerializeToOstream(&output)) {
    std::cerr << "Failed to save to strategy file" << std::endl;
    return false;
  }
  return true;
}


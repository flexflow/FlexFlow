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
#include <fstream>
#include <iostream>
#include <string>

bool load_strategies(std::map<std::string, ParallelConfig>& strategies,
                     const FFProtoBuf::Strategy& strategyPb)
{
  for (int i = 0; i < strategyPb.ops_size(); i++) {
    const FFProtoBuf::Op& op = strategyPb.ops(i);
    ParallelConfig config;
    config.nDims = op.dims_size();
    int n = 1;
    for (int j = 0; j < config.nDims; j++) {
      config.dim[j] = op.dims(j);
      n = n * config.dim[j];
    }
    assert(n == op.devices_size() || op.devices_size() == 0);
    for (int j = 0; j < op.devices_size(); j++)
      config.gpu[j] = op.devices(j);
    strategies[op.name()] = config;
  }
  return true;
}

bool save_strategies(const std::map<std::string, ParallelConfig>& strategies,
                     FFProtoBuf::Strategy& strategyPb)
{
  std::map<std::string, ParallelConfig>::const_iterator it;
  for (it = strategies.begin(); it != strategies.end(); it++) {
    FFProtoBuf::Op* op = strategyPb.add_ops();
    ParallelConfig config = it->second;
    op->set_name(it->first);
    int n = 1;
    for (int j = 0; j < config.nDims; j++) {
      n = n * config.dim[j];
      op->add_dims(config.dim[j]);
    }
    for (int j = 0; j < n; j++) {
      op->add_devices(config.gpu[j]);
    }
  }
  return true;
}

bool FFConfig::load_strategy_file(std::string filename)
{
  FFProtoBuf::Strategy strategyPb;
  std::fstream input(filename, std::ios::in);
  if (!strategyPb.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse strategy file" << std::endl;
    return false;
  }

  load_strategies(strategies, strategyPb);
  printf("strategies.size() = %zu\n", strategies.size());
  return true;
}

bool FFConfig::save_strategy_file(std::string filename)
{
  FFProtoBuf::Strategy strategyPb;
  save_strategies(strategies, strategyPb);
  std::fstream output(filename, std::ios::out | std::ios::trunc);
  if (!strategyPb.SerializeToOstream(&output)) {
    std::cerr << "Failed to save to strategy file" << std::endl;
    return false;
  }
  return true;
}


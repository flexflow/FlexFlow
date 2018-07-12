/* Copyright 2018 Stanford
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

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "model.h"

using namespace std;
using namespace boost::filesystem;

FFModel::FFModel(FFConfig& config)
{
  // Build training dataset
  if (config.datasetPath.length() == 0) {
    config.syntheticInput = true;
  } else {
    path dataset(config.datasetPath);
    std::vector<string> filenames;
    int idx = 0;
    if (is_directory(dataset)) {
      for (auto&entry : boost::make_iterator_range(directory_iterator(dataset), {})) {
        if (is_directory(entry.path())) {
          for (auto&entry2 : boost::make_iterator_range(directory_iterator(entry.path()), {})) {
            
          }
        }
      }
    }
  }
}

void Op::prefetch(const FFModel& ff)
{
  // TODO: perform prefetch for performance imporvement
}

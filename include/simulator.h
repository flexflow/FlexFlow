/* Copyright 2020 Stanford
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
#ifndef _FLEXFLOW_SIMULATOR_H_
#define _FLEXFLOW_SIMULATOR_H_

#include "ffconst.h"
#include "config.h"
class Conv2DMeta;
class LinearMeta;

class Simulator {
public:
  Simulator(FFHandler handle, void* base_ptr, size_t capacity);
  void free_all();
  void* allocate(size_t num_elements, DataType type); 
public:
  FFHandler handle;
  char* base_ptr;
  size_t capacity;
  off_t offset;
  int warmup_times, repeat_times;
  cudaEvent_t start_event, end_event;
public:
  Conv2DMeta* conv2d_meta;
  LinearMeta* linear_meta;
};
#endif

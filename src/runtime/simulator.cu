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

#include "simulator.h"
#include "model.h"

Simulator::Simulator(FFHandler _handle, void* _base_ptr, size_t _capacity)
: handle(_handle), base_ptr((char*)_base_ptr), capacity(_capacity), offset(0),
warmup_times(5), repeat_times(10)
{
  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);
  conv2d_meta = new Conv2DMeta(handle);
  linear_meta = new LinearMeta(handle, 4096);
}

void Simulator::free_all()
{
  offset = 0;
}

void* Simulator::allocate(size_t num_elements, DataType type)
{
  size_t element_size = 0;
  switch (type) {
    case DT_FLOAT:
      element_size = sizeof(float);
      break;
    case DT_DOUBLE:
      element_size = sizeof(double);
      break;
    case DT_INT32:
      element_size = sizeof(int32_t);
      break;
    case DT_INT64:
      element_size = sizeof(int64_t);
      break;
    case DT_BOOLEAN:
      element_size = sizeof(bool);
      break;
    default:
      assert(false);
  }
  void* ret_ptr = base_ptr + offset;
  offset += element_size * num_elements;
  return ret_ptr;
}


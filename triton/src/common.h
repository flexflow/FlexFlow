/* Copyright 2022 NVIDIA CORPORATION
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

#ifndef __LEGION_TRITON_COMMON_H__
#define __LEGION_TRITON_COMMON_H__

#include <string>
#include "triton/core/tritonserver.h"
#include "types.h"

namespace triton { namespace backend { namespace legion {

DataType
ToDataType(const TRITONSERVER_DataType type)
{
  switch (type) {
    case TRITONSERVER_TYPE_FP16:
      return DT_HALF;
    case TRITONSERVER_TYPE_FP32:
      return DT_FLOAT;
    case TRITONSERVER_TYPE_FP64:
      return DT_DOUBLE;
    case TRITONSERVER_TYPE_INT8:
      return DT_INT8;
    case TRITONSERVER_TYPE_INT16:
      return DT_INT16;
    case TRITONSERVER_TYPE_INT32:
      return DT_INT32;
    case TRITONSERVER_TYPE_INT64:
      return DT_INT64;
    case TRITONSERVER_TYPE_UINT8:
      return DT_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return DT_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return DT_UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return DT_UINT64;
    case TRITONSERVER_TYPE_BOOL:
      return DT_BOOLEAN;
    case TRITONSERVER_TYPE_INVALID:
    case TRITONSERVER_TYPE_BYTES:
    default:
      return DT_NONE;
  }
}

TRITONSERVER_DataType
ToTritonDataType(const DataType type)
{
  switch (type) {
    case DT_HALF:
      return TRITONSERVER_TYPE_FP16;
    case DT_FLOAT:
      return TRITONSERVER_TYPE_FP32;
    case DT_DOUBLE:
      return TRITONSERVER_TYPE_FP64;
    case DT_INT8:
      return TRITONSERVER_TYPE_INT8;
    case DT_INT16:
      return TRITONSERVER_TYPE_INT16;
    case DT_INT32:
      return TRITONSERVER_TYPE_INT32;
    case DT_INT64:
      return TRITONSERVER_TYPE_INT64;
    case DT_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case DT_UINT16:
      return TRITONSERVER_TYPE_UINT16;
    case DT_UINT32:
      return TRITONSERVER_TYPE_UINT32;
    case DT_UINT64:
      return TRITONSERVER_TYPE_UINT64;
    case DT_BOOLEAN:
      return TRITONSERVER_TYPE_BOOL;
    case DT_NONE:
    default:
      return TRITONSERVER_TYPE_INVALID;
  }
}

std::string
DataTypeString(const DataType type)
{
  switch (type) {
    case DT_HALF:
      return "DT_HALF";
    case DT_FLOAT:
      return "DT_FLOAT";
    case DT_DOUBLE:
      return "DT_DOUBLE";
    case DT_INT8:
      return "DT_INT8";
    case DT_INT16:
      return "DT_INT16";
    case DT_INT32:
      return "DT_INT32";
    case DT_INT64:
      return "DT_INT64";
    case DT_UINT8:
      return "DT_UINT8";
    case DT_UINT16:
      return "DT_UINT16";
    case DT_UINT32:
      return "DT_UINT32";
    case DT_UINT64:
      return "DT_UINT64";
    case DT_BOOLEAN:
      return "DT_BOOLEAN";
    case DT_NONE:
    default:
      return "DT_NONE";
  }
}

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_COMMON_H__

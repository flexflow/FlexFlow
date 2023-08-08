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

#include <string>
#include "triton/core/tritonserver.h"

namespace {

//
// Duplication of TRITONSERVER_Error implementation
//
class TritonServerError {
 public:
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const char* msg);

  static const char* CodeString(const TRITONSERVER_Error_Code code);
  TRITONSERVER_Error_Code Code() const { return code_; }
  const std::string& Message() const { return msg_; }

 private:
  TritonServerError(TRITONSERVER_Error_Code code, const std::string& msg)
      : code_(code), msg_(msg)
  {
  }
  TritonServerError(TRITONSERVER_Error_Code code, const char* msg)
      : code_(code), msg_(msg)
  {
  }

  TRITONSERVER_Error_Code code_;
  const std::string msg_;
};

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

const char*
TritonServerError::CodeString(const TRITONSERVER_Error_Code code)
{
  switch (code) {
    case TRITONSERVER_ERROR_UNKNOWN:
      return "Unknown";
    case TRITONSERVER_ERROR_INTERNAL:
      return "Internal";
    case TRITONSERVER_ERROR_NOT_FOUND:
      return "Not found";
    case TRITONSERVER_ERROR_INVALID_ARG:
      return "Invalid argument";
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return "Unavailable";
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return "Unsupported";
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      return "Already exists";
    default:
      break;
  }

  return "<invalid code>";
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

TRITONSERVER_Error*
TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      TritonServerError::Create(code, msg));
}

void
TRITONSERVER_ErrorDelete(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  delete lerror;
}

TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Code();
}

const char*
TRITONSERVER_ErrorCodeString(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return TritonServerError::CodeString(lerror->Code());
}

const char*
TRITONSERVER_ErrorMessage(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Message().c_str();
}

#ifdef __cplusplus
}
#endif

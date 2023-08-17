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

#ifndef __LEGION_TRITON_CONFIG_H__
#define __LEGION_TRITON_CONFIG_H__

// Configuration constants for upper bounds for some static properties

// Maximum number of instances per model that we expect to see
#define MAX_NUM_INSTANCES 8

// Maximum number of local processors that we need to handle in this process
#define MAX_LOCAL_PROCS 16

#endif  // __LEGION_TRITON_CONFIG_H__

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

#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include "legion.h"

using namespace Legion;

class FFModel;
class Tensor;

class Initializer
{
public:
  Initializer(void);
  virtual ~Initializer(void);
  virtual void init(Context ctx, Runtime* runtime, const Tensor* tensor) = 0;
};

class GlorotUniform : public Initializer
{
public:
  GlorotUniform(int _seed);
  ~GlorotUniform(void);
  void init(Context ctx, Runtime* runtime, const Tensor* tensor);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
  int seed;
};

class ZeroInitializer : public Initializer
{
public:
  ZeroInitializer(void);
  ~ZeroInitializer(void);
  void init(Context ctx, Runtime* runtime, const Tensor* tensor);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
  static void init_task_cpu(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
};

class UniformInitializer : public Initializer
{
public:
  UniformInitializer(int _seed, float _min, float _max);
  ~UniformInitializer(void);
  void init(Context ctx, Runtime* runtime, const Tensor* tensor);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion>& regions,
                        Context ctx, Runtime *runtime);
  int seed;
  float min_val, max_val;
};

class NormInitializer : public Initializer
{
public:
  NormInitializer(int _seed, float _mean, float _stddev);
  ~NormInitializer(void);
  void init(Context ctx, Runtime* runtime, const Tensor* tensor);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
  int seed;
  float mean, stddev;
};

class ConstantInitializer : public Initializer
{
public:
  ConstantInitializer(float _value);
  ~ConstantInitializer(void);
  void init(Context ctx, Runtime* runtime, const Tensor* tensor);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime);
  static void init_task_cpu(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
public:
  float value;
};
#endif

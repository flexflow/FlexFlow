/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "initializer.h"
#include "model.h"

Initializer::Initializer(void)
{}

Initializer::~Initializer(void)
{}

GlorotUniform::GlorotUniform(void)
: Initializer() {}

GlorotUniform::~GlorotUniform(void)
{}

void GlorotUniform::init(Context ctx,
                         Runtime* runtime,
                         const Tensor* p)
{
  assert(p->numDim == 2);
  int num = std::rand();
  TaskLauncher launcher(GLOROT_INIT_TASK_ID, TaskArgument(&num, sizeof(int)));
  // regions[0]: p->region
  launcher.add_region_requirement(
      RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

ZeroInitializer::ZeroInitializer(void)
: Initializer() 
{}

ZeroInitializer::~ZeroInitializer(void)
{}

void ZeroInitializer::init(Context ctx,
                           Runtime* runtime,
                           const Tensor* p)
{
  TaskLauncher launcher(ZERO_INIT_TASK_ID, TaskArgument(NULL, 0));
  // regions[0]: p->region
  launcher.add_region_requirement(
      RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

UniformInitializer::UniformInitializer(int _seed, float _min, float _max)
: seed(_seed), min_val(_min), max_val(_max) {}

UniformInitializer::~UniformInitializer(void)
{}

void UniformInitializer::init(Context ctx,
                              Runtime* runtime,
                              const Tensor* p)
{
  TaskLauncher launcher(UNIFORM_INIT_TASK_ID,
                        TaskArgument(this, sizeof(UniformInitializer)));
  // regions[0]: p->region
  launcher.add_region_requirement(
      RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

NormInitializer::NormInitializer(int _seed, float _mean, float _stddev)
: seed(_seed), mean(_mean), stddev(_stddev) {}

NormInitializer::~NormInitializer(void)
{}

void NormInitializer::init(Context ctx,
                           Runtime* runtime,
                           const Tensor* p)
{
  TaskLauncher launcher(NORMAL_INIT_TASK_ID,
                        TaskArgument(this, sizeof(NormInitializer)));
  // regions[0]: p->region
  launcher.add_region_requirement(
      RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}


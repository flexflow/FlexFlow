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

#include "strategy.h"
#include <fstream>
#include <iostream>
#include <string>
#include "model.h"
#include "runtime.h"

using namespace Legion;
using namespace Legion::Mapping;

namespace triton { namespace backend { namespace legion {

Logger log_triton("triton");

ShardingFunction::ShardingFunction(ShardingID sid, const LayerStrategy* s)
    : sharding_id(sid), strategy(s)
{
}

Processor
ShardingFunction::find_proc(const DomainPoint& point, const Domain& full_space)
{
  size_t offset = 0;
  const int dims = point.get_dim();
  // We transposed the dimensions when loading the partitioning
  // strategy file, so we need to transpose what order we walk
  // the dimensions when looking for the processor
  assert(dims > 0);
  for (int d = dims - 1; d >= 0; d--) {
    size_t pitch = full_space.hi()[d] - full_space.lo()[d] + 1;
    offset = offset * pitch + point[d] - full_space.lo()[d];
  }
  assert(offset < strategy->global_processors.size());
  return strategy->global_processors[offset];
}

ShardID
ShardingFunction::shard(
    const DomainPoint& point, const Domain& full_space,
    const size_t total_shards)
{
  const Processor proc = find_proc(point, full_space);
  return proc.address_space();
}

LayerStrategy::LayerStrategy(ShardingID sid, MappingTagID t, Runtime* runtime)
    : sharding_function(new ShardingFunction(sid, this)), tag(t)
{
  // Register this sharding functor with legion
  runtime->register_sharding_functor(
      sid, sharding_function, true /*silence warnings*/);
}

LayerStrategy::~LayerStrategy(void)
{
  // TODO: tell legion to unregister the sharding function
}

Domain
LayerStrategy::get_launch_domain(void) const
{
  DomainPoint lo, hi;
  lo.dim = nDims;
  hi.dim = nDims;
  for (int i = 0; i < nDims; i++) {
    lo[i] = 0;
    // Legion domains are inclusive
    assert(dim[i] > 0);
    hi[i] = dim[i] - 1;
  }
  return Domain(lo, hi);
}

Domain
LayerStrategy::find_local_domain(
    Processor proc, const Legion::Domain& global) const
{
  const DomainPoint local_point = find_local_point(proc);
  const int dims = local_point.get_dim();
  assert(dims == global.get_dim());
  DomainPoint lo, hi;
  lo.dim = dims;
  hi.dim = dims;
  for (int d = 0; d < dims; d++) {
    // this will round up so we tile the entire space
    const coord_t tile = (global.hi()[d] - global.lo()[d] + dim[d]) / dim[d];
    lo[d] = local_point[d] * tile;
    hi[d] = (local_point[d] + 1) * tile - 1;
    // clamp to the upper bound space
    if (hi[d] > global.hi()[d])
      hi[d] = global.hi()[d];
  }
  return Domain(lo, hi);
}

bool
LayerStrategy::is_local_processor(Processor proc) const
{
  for (unsigned idx = 0; idx < nProcs; idx++)
    if (local_processors[idx] == proc)
      return true;
  return false;
}

unsigned
LayerStrategy::find_local_offset(Processor proc) const
{
  for (unsigned idx = 0; idx < nProcs; idx++)
    if (local_processors[idx] == proc)
      return idx;
  abort();
  return 0;
}

DomainPoint
LayerStrategy::find_local_point(Realm::Processor proc) const
{
  for (unsigned idx = 0; idx < nProcs; idx++)
    if (local_processors[idx] == proc)
      return local_points[idx];
  abort();
  return DomainPoint();
}

/*static*/ PartitionStrategy*
PartitionStrategy::LoadStrategy(
    const std::string& filename, LegionModelState* model)
{
  std::fstream input(filename, std::ios::in);
  if (!input) {
    std::cerr << "Failed to open strategy file for reading" << std::endl;
    abort();
  }

  int ops_size = 0;
  input >> ops_size;

  LegionTritonRuntime* runtime = model->runtime_;
  // Allocate sharding function IDs for this model
  // We generate a unique string name for this model by concatenating
  // its name with its version number
  const std::string unique_name = model->name + std::to_string(model->version);
  ShardingID first_id = runtime->legion_->generate_library_sharding_ids(
      unique_name.c_str(), ops_size);
  std::vector<const LayerStrategy*> layers(ops_size);
  for (int i = 0; i < ops_size; i++) {
    LayerStrategy* layer = new LayerStrategy(first_id + i, i, runtime->legion_);
    char op_name[64];  // hard-coded size from flexflow
    input >> op_name;
    int device_type;
    input >> device_type;
    switch (device_type) {
      // These are hard-coded from FlexFlow source code
      case 0:
        layer->kind = Processor::TOC_PROC;
        break;
      case 1:
        layer->kind = Processor::LOC_PROC;
        break;
      default:
        fprintf(stderr, "Unsupported Device Type %d\n", device_type);
        abort();
    }
    input >> layer->nDims;
    assert(layer->nDims > 0);
    int n = 1;
    // Note: we transpose the dimensions here from how FlexFlow represents
    // them because we keep our dimensions aligned with ONNX, e.g. NCHW
    for (int j = layer->nDims - 1; j >= 0; j--) {
      input >> layer->dim[j];
      n = n * layer->dim[j];
    }
    int device_ids_size = 0;
    input >> device_ids_size;
    assert(n == device_ids_size || device_ids_size == 0);
    const std::vector<Processor>& all_procs =
        runtime->FindAllProcessors(layer->kind);
    layer->nProcs = 0;
    layer->global_processors.resize(device_ids_size);
    for (int j = 0; j < device_ids_size; j++) {
      int device_id;
      input >> device_id;
      assert(device_id >= 0);
      if (unsigned(device_id) >= all_procs.size()) {
        const char* proc_names[] = {
#define PROC_NAMES(name, desc) desc,
            REALM_PROCESSOR_KINDS(PROC_NAMES)
#undef MEM_NAMES
        };
        std::cerr << "Insufficient " << proc_names[layer->kind]
                  << " processors for partitioning strategy " << filename
                  << std::endl;
        abort();
      }
      const Processor proc = all_procs[device_id];
      layer->global_processors[j] = proc;
      // check to see if it is a local processor
      if (proc.address_space() == runtime->rank_) {
        assert(layer->nProcs < MAX_LOCAL_PROCS);
        layer->local_processors[layer->nProcs++] = proc;
      }
    }
    // Sanity check, compute the mapping of points in the launch domain
    // to local processors so that we can easily invert them later
    Domain launch_domain = layer->get_launch_domain();
    ShardingFunction* function = layer->sharding_function;
    unsigned found_count = 0;
    for (Domain::DomainPointIterator itr(launch_domain); itr; itr++) {
      const Processor proc = function->find_proc(itr.p, launch_domain);
      if (proc.address_space() != runtime->rank_)
        continue;
      bool found = false;
      for (unsigned idx = 0; idx < layer->nProcs; idx++) {
        if (layer->local_processors[idx] != proc)
          continue;
        layer->local_points[idx] = itr.p;
        found = true;
        break;
      }
      assert(found);
      found_count++;
    }
    // Should have found all of them
    assert(found_count == layer->nProcs);
    layers[i] = layer;
  }
  input.close();
  return new PartitionStrategy(model, std::move(layers));
}

PartitionStrategy::~PartitionStrategy(void)
{
  for (auto layer : layers) delete layer;
}

StrategyMapper::StrategyMapper(
    const PartitionStrategy* s, Mapping::MapperRuntime* rt, Machine m)
    : Mapper(rt), strategy(s), machine(m), local_node(get_local_node()),
      total_nodes(get_total_nodes(m)), mapper_name(create_name(local_node))
{
  // Query to find all our local processors
  Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  for (Machine::ProcessorQuery::iterator it = local_procs.begin();
       it != local_procs.end(); it++) {
    switch (it->kind()) {
      case Processor::LOC_PROC: {
        local_cpus.push_back(*it);
        break;
      }
      case Processor::TOC_PROC: {
        local_gpus.push_back(*it);
        break;
      }
      case Processor::OMP_PROC: {
        local_omps.push_back(*it);
        break;
      }
      case Processor::IO_PROC: {
        local_ios.push_back(*it);
        break;
      }
      case Processor::PY_PROC: {
        local_pys.push_back(*it);
        break;
      }
      default:
        break;
    }
  }
  // Now do queries to find all our local memories
  Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() > 0);
  local_system_memory = local_sysmem.first();
  if (!local_gpus.empty()) {
    Machine::MemoryQuery local_zcmem(machine);
    local_zcmem.local_address_space();
    local_zcmem.only_kind(Memory::Z_COPY_MEM);
    assert(local_zcmem.count() > 0);
    local_zerocopy_memory = local_zcmem.first();
  }
  for (std::vector<Processor>::const_iterator it = local_gpus.begin();
       it != local_gpus.end(); it++) {
    Machine::MemoryQuery local_framebuffer(machine);
    local_framebuffer.local_address_space();
    local_framebuffer.only_kind(Memory::GPU_FB_MEM);
    local_framebuffer.best_affinity_to(*it);
    assert(local_framebuffer.count() > 0);
    local_frame_buffers[*it] = local_framebuffer.first();
  }
  for (std::vector<Processor>::const_iterator it = local_omps.begin();
       it != local_omps.end(); it++) {
    Machine::MemoryQuery local_numa(machine);
    local_numa.local_address_space();
    local_numa.only_kind(Memory::SOCKET_MEM);
    local_numa.best_affinity_to(*it);
    if (local_numa.count() > 0)  // if we have NUMA memories then use them
      local_numa_domains[*it] = local_numa.first();
    else  // Otherwise we just use the local system memory
      local_numa_domains[*it] = local_system_memory;
  }
}

StrategyMapper::~StrategyMapper(void)
{
  free(const_cast<char*>(mapper_name));
}

//--------------------------------------------------------------------------
/*static*/ AddressSpace
StrategyMapper::get_local_node(void)
//--------------------------------------------------------------------------
{
  Processor p = Processor::get_executing_processor();
  return p.address_space();
}

//--------------------------------------------------------------------------
/*static*/ size_t
StrategyMapper::get_total_nodes(Machine m)
//--------------------------------------------------------------------------
{
  Machine::ProcessorQuery query(m);
  query.only_kind(Processor::LOC_PROC);
  std::set<AddressSpace> spaces;
  for (Machine::ProcessorQuery::iterator it = query.begin(); it != query.end();
       it++)
    spaces.insert(it->address_space());
  return spaces.size();
}

//--------------------------------------------------------------------------
/*static*/ const char*
StrategyMapper::create_name(AddressSpace node)
//--------------------------------------------------------------------------
{
  char buffer[128];
  snprintf(buffer, 127, "Legion Triton Mapper on Node %d", node);
  return strdup(buffer);
}

//--------------------------------------------------------------------------
const char*
StrategyMapper::get_mapper_name(void) const
//--------------------------------------------------------------------------
{
  return mapper_name;
}

//--------------------------------------------------------------------------
Mapper::MapperSyncModel
StrategyMapper::get_mapper_sync_model(void) const
//--------------------------------------------------------------------------
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_task_options(
    const MapperContext ctx, const Task& task, TaskOptions& output)
//--------------------------------------------------------------------------
{
  assert(task.get_depth() > 0);
  if (!local_gpus.empty() && has_variant(ctx, task, Processor::TOC_PROC))
    output.initial_proc = local_gpus.front();
  else if (!local_omps.empty() && has_variant(ctx, task, Processor::OMP_PROC))
    output.initial_proc = local_omps.front();
  else
    output.initial_proc = local_cpus.front();
  // We never want valid instances
  output.valid_instances = false;
}

//--------------------------------------------------------------------------
void
StrategyMapper::premap_task(
    const MapperContext ctx, const Task& task, const PremapTaskInput& input,
    PremapTaskOutput& output)
//--------------------------------------------------------------------------
{
  // NO-op since we know that all our futures should be mapped in the system
  // memory
}

//--------------------------------------------------------------------------
void
StrategyMapper::slice_task(
    const MapperContext ctx, const Task& task, const SliceTaskInput& input,
    SliceTaskOutput& output)
//--------------------------------------------------------------------------
{
  // For multi-node cases we should already have been sharded so we
  // should just have one or a few points here on this node, so iterate
  // them and round-robin them across the local processors here
  output.slices.reserve(input.domain.get_volume());
  // Get the sharding functor for this operation and then use it to localize
  // the points onto the processors of this shard
  ShardingFunction* function = find_sharding_functor(task);
  // Get the domain for the sharding space also
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);
  switch (function->strategy->kind) {
    case Processor::LOC_PROC: {
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        const Processor proc = function->find_proc(itr.p, sharding_domain);
        assert(proc.kind() == Processor::LOC_PROC);
        output.slices.push_back(TaskSlice(
            Domain(itr.p, itr.p), proc, false /*recurse*/,
            false /*stealable*/));
      }
      break;
    }
    case Processor::TOC_PROC: {
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        const Processor proc = function->find_proc(itr.p, sharding_domain);
        assert(proc.kind() == Processor::TOC_PROC);
        output.slices.push_back(TaskSlice(
            Domain(itr.p, itr.p), proc, false /*recurse*/,
            false /*stealable*/));
      }
      break;
    }
    case Processor::OMP_PROC: {
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        const Processor proc = function->find_proc(itr.p, sharding_domain);
        assert(proc.kind() == Processor::OMP_PROC);
        output.slices.push_back(TaskSlice(
            Domain(itr.p, itr.p), proc, false /*recurse*/,
            false /*stealable*/));
      }
      break;
    }
    default:
      abort();
  }
}

//--------------------------------------------------------------------------
bool
StrategyMapper::has_variant(
    const MapperContext ctx, const Task& task, Processor::Kind kind)
//--------------------------------------------------------------------------
{
  const std::pair<TaskID, Processor::Kind> key(task.task_id, kind);
  // Check to see if we already have it
  std::map<std::pair<TaskID, Processor::Kind>, VariantID>::const_iterator
      finder = used_variants.find(key);
  if ((finder != used_variants.end()) && (finder->second != 0))
    return true;
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, key.first, variants, key.second);
  assert(variants.size() <= 1);
  if (variants.empty())
    return false;
  used_variants[key] = variants.front();
  return true;
}

//--------------------------------------------------------------------------
VariantID
StrategyMapper::find_variant(const MapperContext ctx, const Task& task)
//--------------------------------------------------------------------------
{
  return find_variant(ctx, task, task.target_proc);
}

//--------------------------------------------------------------------------
VariantID
StrategyMapper::find_variant(
    const MapperContext ctx, const Task& task, Processor target_proc)
//--------------------------------------------------------------------------
{
  const std::pair<TaskID, Processor::Kind> key(
      task.task_id, target_proc.kind());
  std::map<std::pair<TaskID, Processor::Kind>, VariantID>::const_iterator
      finder = used_variants.find(key);
  if ((finder != used_variants.end()) && (finder->second != 0))
    return finder->second;
  // Haven't seen it before so let's look it up to make sure it exists
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, key.first, variants, key.second);
  assert(variants.size() <= 1);
  if (variants.empty())
    log_triton.error(
        "Unable to find variant for task %s to run on processor %llx.",
        task.get_task_name(), target_proc.id);
  VariantID result = variants.front();
  used_variants[key] = result;
  return result;
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_task(
    const MapperContext ctx, const Task& task, const MapTaskInput& input,
    MapTaskOutput& output)
//--------------------------------------------------------------------------
{
  // Should never be mapping the top-level task here
  assert(task.get_depth() > 0);
  // This is one of our normal operator tasks
  // First let's see if this is sub-rankable
  output.chosen_instances.resize(task.regions.size());
  output.chosen_variant = find_variant(ctx, task);
  // Normal task and not sub-rankable, so let's actually do the mapping
  Memory target_memory = Memory::NO_MEMORY;
  switch (task.target_proc.kind()) {
    case Processor::LOC_PROC: {
      target_memory = local_system_memory;
      break;
    }
    case Processor::TOC_PROC: {
      target_memory = local_frame_buffers[task.target_proc];
      break;
    }
    case Processor::OMP_PROC: {
      target_memory = local_numa_domains[task.target_proc];
      break;
    }
    default:
      abort();
  }
  // Map each field separately for each of the logical regions
  std::vector<PhysicalInstance> needed_acquires;
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    const RegionRequirement& req = task.regions[idx];
    // Skip any regions that have been projected out
    if (!req.region.exists())
      continue;
    std::vector<PhysicalInstance>& instances = output.chosen_instances[idx];
    // Get the reference to our valid instances in case we decide to use them
    const std::vector<PhysicalInstance>& valid = input.valid_instances[idx];
    instances.resize(req.privilege_fields.size());
    unsigned index = 0;
    for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
         it != req.privilege_fields.end(); it++, index++)
      if (map_tensor(
              ctx, task, idx, req.region, *it, target_memory, task.target_proc,
              valid, instances[index], req.redop))
        needed_acquires.push_back(instances[index]);
  }
  // Do an acquire on all the instances so we have our result
  // Keep doing this until we succed or we get an out of memory error
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(
             ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    // If we failed to acquire any of the instances we need to prune them
    // out of the mapper's data structure so do that first
    std::set<PhysicalInstance> failed_acquires;
    filter_failed_acquires(needed_acquires, failed_acquires);
    // Now go through all our region requirements and and figure out which
    // region requirements and fields need to attempt to remap
    for (unsigned idx1 = 0; idx1 < task.regions.size(); idx1++) {
      const RegionRequirement& req = task.regions[idx1];
      // Skip any regions that have been projected out
      if (!req.region.exists())
        continue;
      std::vector<PhysicalInstance>& instances = output.chosen_instances[idx1];
      std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
      for (unsigned idx2 = 0; idx2 < instances.size(); idx2++, fit++) {
        if (failed_acquires.find(instances[idx2]) == failed_acquires.end())
          continue;
        // Now try to remap it
        const FieldID fid = *fit;
        const std::vector<PhysicalInstance>& valid =
            input.valid_instances[idx1];
        if (map_tensor(
                ctx, task, idx1, req.region, fid, target_memory,
                task.target_proc, valid, instances[idx2], req.redop))
          needed_acquires.push_back(instances[idx2]);
      }
    }
  }
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_replicate_task(
    const MapperContext ctx, const Task& task, const MapTaskInput& input,
    const MapTaskOutput& def_output, MapReplicateTaskOutput& output)
//--------------------------------------------------------------------------
{
  abort();
}

//--------------------------------------------------------------------------
bool
StrategyMapper::find_existing_instance(
    LogicalRegion region, FieldID fid, Memory target_memory,
    PhysicalInstance& result)
//--------------------------------------------------------------------------
{
  // See if we already have it in our local instances
  const FieldMemInfo info(region.get_tree_id(), fid, target_memory);
  std::map<FieldMemInfo, InstanceInfos>::const_iterator finder =
      local_instances.find(info);
  if ((finder != local_instances.end()) &&
      finder->second.has_instance(region, result))
    return true;
  // See if we can find an existing instance in any memory
  const FieldMemInfo info_sysmem(
      region.get_tree_id(), fid, local_system_memory);
  finder = local_instances.find(info_sysmem);
  if ((finder != local_instances.end()) &&
      finder->second.has_instance(region, result)) {
    return true;
  }
  for (std::map<Processor, Memory>::const_iterator it =
           local_frame_buffers.begin();
       it != local_frame_buffers.end(); it++) {
    const FieldMemInfo info_fb(region.get_tree_id(), fid, it->second);
    finder = local_instances.find(info_fb);
    if ((finder != local_instances.end()) &&
        finder->second.has_instance(region, result)) {
      return true;
    }
  }
  for (std::map<Processor, Memory>::const_iterator it =
           local_numa_domains.begin();
       it != local_numa_domains.end(); it++) {
    const FieldMemInfo info_numa(region.get_tree_id(), fid, it->second);
    finder = local_instances.find(info_numa);
    if ((finder != local_instances.end()) &&
        finder->second.has_instance(region, result)) {
      return true;
    }
  }
  return false;
}

//--------------------------------------------------------------------------
bool
StrategyMapper::map_tensor(
    const MapperContext ctx, const Mappable& mappable, unsigned index,
    LogicalRegion region, FieldID fid, Memory target_memory,
    Processor target_proc, const std::vector<PhysicalInstance>& valid,
    PhysicalInstance& result, ReductionOpID redop /*=0*/)
//--------------------------------------------------------------------------
{
  // If we're making a reduction instance, we should just make it now
  if (redop != 0) {
    // Switch the target memory if we're going to a GPU because
    // Realm's DMA system still does not support reductions
    if (target_memory.kind() == Memory::GPU_FB_MEM)
      target_memory = local_zerocopy_memory;
    const std::vector<LogicalRegion> regions(1, region);
    LayoutConstraintSet layout_constraints;
    // No specialization
    layout_constraints.add_constraint(
        SpecializedConstraint(REDUCTION_FOLD_SPECIALIZE, redop));
    // SOA-C dimension ordering
    std::vector<DimensionKind> dimension_ordering(4);
    dimension_ordering[0] = DIM_Z;
    dimension_ordering[1] = DIM_Y;
    dimension_ordering[2] = DIM_X;
    dimension_ordering[3] = DIM_F;
    layout_constraints.add_constraint(
        OrderingConstraint(dimension_ordering, false /*contiguous*/));
    // Constraint for the kind of memory
    layout_constraints.add_constraint(MemoryConstraint(target_memory.kind()));
    // Make sure we have our field
    const std::vector<FieldID> fields(1, fid);
    layout_constraints.add_constraint(
        FieldConstraint(fields, true /*contiguous*/));
    if (!runtime->create_physical_instance(
            ctx, target_memory, layout_constraints, regions, result,
            true /*acquire*/))
      report_failed_mapping(mappable, index, target_memory, redop);
    // We already did the acquire
    return false;
  }
  // See if we already have it in our local instances
  const FieldMemInfo info_key(region.get_tree_id(), fid, target_memory);
  std::map<FieldMemInfo, InstanceInfos>::const_iterator finder =
      local_instances.find(info_key);
  if ((finder != local_instances.end()) &&
      finder->second.has_instance(region, result)) {
    // Needs acquire to keep the runtime happy
    return true;
  }
  // There's a little asymmetry here between CPUs and GPUs for NUMA effects
  // For CPUs NUMA-effects are within a factor of 2X additional latency and
  // reduced bandwidth, so it's better to just use data where it is rather
  // than move it. For GPUs though, the difference between local framebuffer
  // and remote can be on the order of 800 GB/s versus 20 GB/s over NVLink
  // so it's better to move things local, so we'll always try to make a local
  // instance before checking for a nearby instance in a different GPU.
  if (target_proc.exists() && ((target_proc.kind() == Processor::LOC_PROC) ||
                               (target_proc.kind() == Processor::OMP_PROC))) {
    Machine::MemoryQuery affinity_mems(machine);
    affinity_mems.has_affinity_to(target_proc);
    for (Machine::MemoryQuery::iterator it = affinity_mems.begin();
         it != affinity_mems.end(); it++) {
      const FieldMemInfo affinity_info(region.get_tree_id(), fid, *it);
      finder = local_instances.find(affinity_info);
      if ((finder != local_instances.end()) &&
          finder->second.has_instance(region, result))
        // Needs acquire to keep the runtime happy
        return true;
    }
  }
  // Haven't made this instance before, so make it now
  // We can do an interesting optimization here to try to reduce unnecessary
  // inter-memory copies. For logical regions that are overlapping we try
  // to accumulate as many as possible into one physical instance and use
  // that instance for all the tasks for the different regions.
  // First we have to see if there is anything we overlap with
  const IndexSpace is = region.get_index_space();
  // This whole process has to appear atomic
  runtime->disable_reentrant(ctx);
  InstanceInfos& infos = local_instances[info_key];
  // One more check once we get the lock
  if (infos.has_instance(region, result)) {
    runtime->enable_reentrant(ctx);
    return true;
  }
  const Domain dom = runtime->get_index_space_domain(ctx, is);
  std::vector<unsigned> overlaps;
  // Regions to include in the overlap from other fields
  std::set<LogicalRegion> other_field_overlaps;
  // This is guaranteed to be a rectangle
  Domain upper_bound;
  switch (is.get_dim()) {
#define DIMFUNC(DN)                                                          \
  case DN: {                                                                 \
    bool changed = false;                                                    \
    Rect<DN> bound = dom.bounds<DN, coord_t>();                              \
    for (unsigned idx = 0; idx < infos.instances.size(); idx++) {            \
      const InstanceInfo& info = infos.instances[idx];                       \
      Rect<DN> other = info.bounding_box;                                    \
      Rect<DN> intersect = bound.intersection(other);                        \
      if (intersect.empty())                                                 \
        continue;                                                            \
      /*Don't merge if the unused space would be more than the space saved*/ \
      Rect<DN> union_bbox = bound.union_bbox(other);                         \
      size_t bound_volume = bound.volume();                                  \
      size_t union_volume = union_bbox.volume();                             \
      /* If it didn't get any bigger then we can keep going*/                \
      if (bound_volume == union_volume)                                      \
        continue;                                                            \
      size_t intersect_volume = intersect.volume();                          \
      /* Only allow merging if it isn't "too big"*/                          \
      /* We define "too big" as the size of the "unused" points being bigger \
       * than the intersection*/                                             \
      if ((union_volume - (bound_volume + other.volume() -                   \
                           intersect_volume)) > intersect_volume)            \
        continue;                                                            \
      overlaps.push_back(idx);                                               \
      bound = union_bbox;                                                    \
      changed = true;                                                        \
    }                                                                        \
    /* If we didn't find any overlapping modifications check adjacent fields \
     * in the same tree*/                                                    \
    /* to see if we can use them to infer what our shape should be.*/        \
    if (!changed) {                                                          \
      for (std::map<FieldMemInfo, InstanceInfos>::const_iterator it =        \
               local_instances.begin();                                      \
           it != local_instances.end(); it++) {                              \
        if ((it->first.tid != info_key.tid) ||                               \
            (it->first.fid == info_key.fid) ||                               \
            (it->first.memory != info_key.memory))                           \
          continue;                                                          \
        std::map<LogicalRegion, unsigned>::const_iterator finder =           \
            it->second.region_mapping.find(region);                          \
        if (finder != it->second.region_mapping.end()) {                     \
          const InstanceInfo& other_info =                                   \
              it->second.instances[finder->second];                          \
          Rect<DN> other = other_info.bounding_box;                          \
          bound = bound.union_bbox(other);                                   \
          other_field_overlaps.insert(                                       \
              other_info.regions.begin(), other_info.regions.end());         \
        }                                                                    \
      }                                                                      \
    }                                                                        \
    upper_bound = Domain(bound);                                             \
    break;                                                                   \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  // We're going to need some of this constraint information no matter
  // which path we end up taking below
  LayoutConstraintSet layout_constraints;
  // No specialization
  layout_constraints.add_constraint(SpecializedConstraint());
  // SOA-C dimension ordering
  std::vector<DimensionKind> dimension_ordering(4);
  dimension_ordering[0] = DIM_Z;
  dimension_ordering[1] = DIM_Y;
  dimension_ordering[2] = DIM_X;
  dimension_ordering[3] = DIM_F;
  layout_constraints.add_constraint(
      OrderingConstraint(dimension_ordering, false /*contiguous*/));
  // Constraint for the kind of memory
  layout_constraints.add_constraint(MemoryConstraint(target_memory.kind()));
  // Make sure we have our field
  const std::vector<FieldID> fields(1, fid);
  layout_constraints.add_constraint(
      FieldConstraint(fields, true /*contiguous*/));
  // Check to see if we have any overlaps
  if (overlaps.empty()) {
    // No overlaps, so just go ahead and make our instance and add it
    std::vector<LogicalRegion> regions(1, region);
    // If we're bringing in other regions include them as well in this set
    if (!other_field_overlaps.empty()) {
      other_field_overlaps.erase(region);
      regions.insert(
          regions.end(), other_field_overlaps.begin(),
          other_field_overlaps.end());
    }
    bool created;
    size_t footprint;
    if (runtime->find_or_create_physical_instance(
            ctx, target_memory, layout_constraints, regions, result, created,
            true /*acquire*/, GC_NEVER_PRIORITY, false /*tight bounds*/,
            &footprint)) {
      // We succeeded in making the instance where we want it
      assert(result.exists());
      if (created)
        log_triton.info(
            "%s created instance %lx containing %zd bytes in memory " IDFMT,
            mapper_name, result.get_instance_id(), footprint, target_memory.id);
      // Only save the result for future use if it is not an external instance
      if (!result.is_external_instance()) {
        const unsigned idx = infos.insert(region, upper_bound, result);
        InstanceInfo& info = infos.instances[idx];
        for (std::set<LogicalRegion>::const_iterator it =
                 other_field_overlaps.begin();
             it != other_field_overlaps.end(); it++) {
          if ((*it) == region)
            continue;
          infos.region_mapping[*it] = idx;
          info.regions.push_back(*it);
        }
      }
      // We made it so no need for an acquire
      runtime->enable_reentrant(ctx);
      return false;
    }

  } else if (overlaps.size() == 1) {
    // Overlap with exactly one other instance
    InstanceInfo& info = infos.instances[overlaps[0]];
    // A Legion bug prevents us from doing this case
    if (info.bounding_box == upper_bound) {
      // Easy case of dominance, so just add it
      info.regions.push_back(region);
      infos.region_mapping[region] = overlaps[0];
      result = info.instance;
      runtime->enable_reentrant(ctx);
      // Didn't make it so we need to acquire it
      return true;
    } else {
      // We have to make a new instance
      info.regions.push_back(region);
      bool created;
      size_t footprint;
      if (runtime->find_or_create_physical_instance(
              ctx, target_memory, layout_constraints, info.regions, result,
              created, true /*acquire*/, GC_NEVER_PRIORITY,
              false /*tight bounds*/, &footprint)) {
        // We succeeded in making the instance where we want it
        assert(result.exists());
        if (created)
          log_triton.info(
              "%s created instance %lx containing %zd bytes in memory " IDFMT,
              mapper_name, result.get_instance_id(), footprint,
              target_memory.id);
        // Remove the GC priority on the old instance back to 0
        runtime->set_garbage_collection_priority(ctx, info.instance, 0);
        // Update everything in place
        info.instance = result;
        info.bounding_box = upper_bound;
        infos.region_mapping[region] = overlaps[0];
        runtime->enable_reentrant(ctx);
        // We made it so no need for an acquire
        return false;
      } else  // Failed to make it so pop the logical region name back off
        info.regions.pop_back();
    }
  } else {
    // Overlap with multiple previous instances
    std::vector<LogicalRegion> combined_regions(1, region);
    for (std::vector<unsigned>::const_iterator it = overlaps.begin();
         it != overlaps.end(); it++)
      combined_regions.insert(
          combined_regions.end(), infos.instances[*it].regions.begin(),
          infos.instances[*it].regions.end());
    // Try to make it
    bool created;
    size_t footprint;
    if (runtime->find_or_create_physical_instance(
            ctx, target_memory, layout_constraints, combined_regions, result,
            created, true /*acquire*/, GC_NEVER_PRIORITY,
            false /*tight bounds*/, &footprint)) {
      // We succeeded in making the instance where we want it
      assert(result.exists());
      if (created)
        log_triton.info(
            "%s created instance %lx containing %zd bytes in memory " IDFMT,
            mapper_name, result.get_instance_id(), footprint, target_memory.id);
      // Remove all the previous entries back to front
      for (std::vector<unsigned>::const_reverse_iterator it =
               overlaps.crbegin();
           it != overlaps.crend(); it++) {
        // Remove the GC priority on the old instance
        runtime->set_garbage_collection_priority(
            ctx, infos.instances[*it].instance, 0);
        infos.instances.erase(infos.instances.begin() + *it);
      }
      // Add the new entry
      const unsigned index = infos.instances.size();
      infos.instances.resize(index + 1);
      InstanceInfo& info = infos.instances[index];
      info.instance = result;
      info.bounding_box = upper_bound;
      info.regions = combined_regions;
      // Update the mappings for all the instances
      // This really sucks but it should be pretty rare
      // We can start at the entry of the first overlap since everything
      // before that is guaranteed to be unchanged
      for (unsigned idx = overlaps[0]; idx < infos.instances.size(); idx++) {
        for (std::vector<LogicalRegion>::const_iterator it =
                 infos.instances[idx].regions.begin();
             it != infos.instances[idx].regions.end(); it++)
          infos.region_mapping[*it] = idx;
      }
      runtime->enable_reentrant(ctx);
      // We made it so no need for an acquire
      return false;
    }
  }
  // Done with the atomic part
  runtime->enable_reentrant(ctx);
  // If we get here it's because we failed to make the instance, we still
  // have a few more tricks that we can try
  // First see if we can find an existing valid instance that we can use
  // with affinity to our target processor
  if (!valid.empty()) {
    for (std::vector<PhysicalInstance>::const_iterator it = valid.begin();
         it != valid.end(); it++) {
      // If it doesn't have the field then we don't care
      if (!it->has_field(fid))
        continue;
      if (!target_proc.exists() ||
          machine.has_affinity(target_proc, it->get_location())) {
        result = *it;
        return true;
      }
    }
  }
  // Still couldn't find an instance, see if we can find any instances
  // in memories that are local to our node that we can use
  if (target_proc.exists()) {
    Machine::MemoryQuery affinity_mems(machine);
    affinity_mems.has_affinity_to(target_proc);
    for (Machine::MemoryQuery::iterator it = affinity_mems.begin();
         it != affinity_mems.end(); it++) {
      const FieldMemInfo affinity_info(region.get_tree_id(), fid, *it);
      finder = local_instances.find(affinity_info);
      if ((finder != local_instances.end()) &&
          finder->second.has_instance(region, result))
        // Needs acquire to keep the runtime happy
        return true;
    }
  } else if (find_existing_instance(region, fid, target_memory, result)) {
    return true;
  }
  // If we make it here then we failed entirely
  report_failed_mapping(mappable, index, target_memory, redop);
  return true;
}

//--------------------------------------------------------------------------
void
StrategyMapper::filter_failed_acquires(
    std::vector<PhysicalInstance>& needed_acquires,
    std::set<PhysicalInstance>& failed_acquires)
//--------------------------------------------------------------------------
{
  for (std::vector<PhysicalInstance>::const_iterator it =
           needed_acquires.begin();
       it != needed_acquires.end(); it++) {
    if (failed_acquires.find(*it) != failed_acquires.end())
      continue;
    failed_acquires.insert(*it);
    const Memory mem = it->get_location();
    const RegionTreeID tid = it->get_tree_id();
    for (std::map<FieldMemInfo, InstanceInfos>::iterator fit =
             local_instances.begin();
         fit != local_instances.end();
         /*nothing*/) {
      if ((fit->first.memory != mem) || (fit->first.tid != tid)) {
        fit++;
        continue;
      }
      if (fit->second.filter(*it)) {
        std::map<FieldMemInfo, InstanceInfos>::iterator to_delete = fit++;
        local_instances.erase(to_delete);
      } else
        fit++;
    }
  }
  needed_acquires.clear();
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_failed_mapping(
    const Mappable& mappable, unsigned index, Memory target_memory,
    ReductionOpID redop)
//--------------------------------------------------------------------------
{
  const char* memory_kinds[] = {
#define MEM_NAMES(name, desc) desc,
      REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
  };
  switch (mappable.get_mappable_type()) {
    case Mappable::TASK_MAPPABLE: {
      const Task* task = mappable.as_task();
      if (redop > 0)
        log_triton.error(
            "Mapper %s failed to map reduction (%d) region "
            "requirement %d of task %s (UID %lld) into %s memory " IDFMT,
            get_mapper_name(), redop, index, task->get_task_name(),
            mappable.get_unique_id(), memory_kinds[target_memory.kind()],
            target_memory.id);
      else
        log_triton.error(
            "Mapper %s failed to map region requirement %d of "
            "task %s (UID %lld) into %s memory " IDFMT,
            get_mapper_name(), index, task->get_task_name(),
            mappable.get_unique_id(), memory_kinds[target_memory.kind()],
            target_memory.id);
      break;
    }
    case Mappable::COPY_MAPPABLE: {
      if (redop > 0)
        log_triton.error(
            "Mapper %s failed to map reduction (%d) region "
            "requirement %d of copy (UID %lld) into %s memory " IDFMT,
            get_mapper_name(), redop, index, mappable.get_unique_id(),
            memory_kinds[target_memory.kind()], target_memory.id);
      else
        log_triton.error(
            "Mapper %s failed to map region requirement %d of "
            "copy (UID %lld) into %s memory " IDFMT,
            get_mapper_name(), index, mappable.get_unique_id(),
            memory_kinds[target_memory.kind()], target_memory.id);
      break;
    }
    case Mappable::INLINE_MAPPABLE: {
      if (redop > 0)
        log_triton.error(
            "Mapper %s failed to map reduction (%d) region "
            "requirement %d of inline mapping (UID %lld) into %s memory " IDFMT,
            get_mapper_name(), redop, index, mappable.get_unique_id(),
            memory_kinds[target_memory.kind()], target_memory.id);
      else
        log_triton.error(
            "Mapper %s failed to map region requirement %d of "
            "inline mapping (UID %lld) into %s memory " IDFMT,
            get_mapper_name(), index, mappable.get_unique_id(),
            memory_kinds[target_memory.kind()], target_memory.id);
      break;
    }
    case Mappable::PARTITION_MAPPABLE: {
      assert(redop == 0);
      log_triton.error(
          "Mapper %s failed to map region requirement %d of "
          "partition (UID %lld) into %s memory " IDFMT,
          get_mapper_name(), index, mappable.get_unique_id(),
          memory_kinds[target_memory.kind()], target_memory.id);
      break;
    }
    default:
      abort();  // should never get here
  }
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_task_variant(
    const MapperContext ctx, const Task& task, const SelectVariantInput& input,
    SelectVariantOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_variant = find_variant(ctx, task, input.processor);
}

//--------------------------------------------------------------------------
void
StrategyMapper::postmap_task(
    const MapperContext ctx, const Task& task, const PostMapInput& input,
    PostMapOutput& output)
//--------------------------------------------------------------------------
{
  // We should currently never get this call in triton
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_task_sources(
    const MapperContext ctx, const Task& task, const SelectTaskSrcInput& input,
    SelectTaskSrcOutput& output)
//--------------------------------------------------------------------------
{
  triton_select_sources(
      ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void
StrategyMapper::triton_select_sources(
    const MapperContext ctx, const PhysicalInstance& target,
    const std::vector<PhysicalInstance>& sources,
    std::deque<PhysicalInstance>& ranking)
//--------------------------------------------------------------------------
{
  std::map<Memory, unsigned /*bandwidth*/> source_memories;
  // For right now we'll rank instances by the bandwidth of the memory
  // they are in to the destination, we'll only rank sources from the
  // local node if there are any
  bool all_local = false;
  // TODO: consider layouts when ranking source to help out the DMA system
  Memory destination_memory = target.get_location();
  std::vector<MemoryMemoryAffinity> affinity(1);
  // fill in a vector of the sources with their bandwidths and sort them
  std::vector<std::pair<PhysicalInstance, unsigned /*bandwidth*/>> band_ranking;
  for (unsigned idx = 0; idx < sources.size(); idx++) {
    const PhysicalInstance& instance = sources[idx];
    Memory location = instance.get_location();
    if (location.address_space() == local_node) {
      if (!all_local) {
        source_memories.clear();
        band_ranking.clear();
        all_local = true;
      }
    } else if (all_local)  // Skip any remote instances once we're local
      continue;
    std::map<Memory, unsigned>::const_iterator finder =
        source_memories.find(location);
    if (finder == source_memories.end()) {
      affinity.clear();
      machine.get_mem_mem_affinity(
          affinity, location, destination_memory,
          false /*not just local affinities*/);
      unsigned memory_bandwidth = 0;
      if (!affinity.empty()) {
        assert(affinity.size() == 1);
        memory_bandwidth = affinity[0].bandwidth;
      }
      source_memories[location] = memory_bandwidth;
      band_ranking.push_back(
          std::pair<PhysicalInstance, unsigned>(instance, memory_bandwidth));
    } else
      band_ranking.push_back(
          std::pair<PhysicalInstance, unsigned>(instance, finder->second));
  }
  assert(!band_ranking.empty());
  // Easy case of only one instance
  if (band_ranking.size() == 1) {
    ranking.push_back(band_ranking.begin()->first);
    return;
  }
  // Sort them by bandwidth
  std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);
  // Iterate from largest bandwidth to smallest
  for (std::vector<
           std::pair<PhysicalInstance, unsigned>>::const_reverse_iterator it =
           band_ranking.rbegin();
       it != band_ranking.rend(); it++)
    ranking.push_back(it->first);
}

//--------------------------------------------------------------------------
void
StrategyMapper::speculate(
    const MapperContext ctx, const Task& task, SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_profiling(
    const MapperContext ctx, const Task& task, const TaskProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // Shouldn't get any profiling feedback currently
  abort();
}


//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const Task& task,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = find_sharding_functor(task)->sharding_id;
}

//--------------------------------------------------------------------------
ShardingFunction*
StrategyMapper::find_sharding_functor(const Mappable& mappable)
//--------------------------------------------------------------------------
{
  assert(mappable.tag < strategy->layers.size());
  return strategy->layers[mappable.tag]->sharding_function;
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_inline(
    const MapperContext ctx, const InlineMapping& inline_op,
    const MapInlineInput& input, MapInlineOutput& output)
//--------------------------------------------------------------------------
{
  const std::vector<PhysicalInstance>& valid = input.valid_instances;
  const RegionRequirement& req = inline_op.requirement;
  output.chosen_instances.resize(req.privilege_fields.size());
  unsigned index = 0;
  std::vector<PhysicalInstance> needed_acquires;
  for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
       it != req.privilege_fields.end(); it++, index++)
    if (map_tensor(
            ctx, inline_op, 0, req.region, *it, local_system_memory,
            inline_op.parent_task->current_proc, valid,
            output.chosen_instances[index], req.redop))
      needed_acquires.push_back(output.chosen_instances[index]);
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(
             ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    std::set<PhysicalInstance> failed_instances;
    filter_failed_acquires(needed_acquires, failed_instances);
    // Now go through all the fields for the instances and try and remap
    std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
    for (unsigned idx = 0; idx < output.chosen_instances.size(); idx++, fit++) {
      if (failed_instances.find(output.chosen_instances[idx]) ==
          failed_instances.end())
        continue;
      // Now try to remap it
      if (map_tensor(
              ctx, inline_op, 0 /*idx*/, req.region, *fit, local_system_memory,
              inline_op.parent_task->current_proc, valid,
              output.chosen_instances[idx]))
        needed_acquires.push_back(output.chosen_instances[idx]);
    }
  }
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_inline_sources(
    const MapperContext ctx, const InlineMapping& inline_op,
    const SelectInlineSrcInput& input, SelectInlineSrcOutput& output)
//--------------------------------------------------------------------------
{
  triton_select_sources(
      ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_profiling(
    const MapperContext ctx, const InlineMapping& inline_op,
    const InlineProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling yet for inline mappings
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_copy(
    const MapperContext ctx, const Copy& copy, const MapCopyInput& input,
    MapCopyOutput& output)
//--------------------------------------------------------------------------
{
  // We should always be able to materialize instances of the things
  // we are copying so make concrete source instances
  std::vector<PhysicalInstance> needed_acquires;
  Memory target_memory = local_system_memory;
  if (copy.is_index_space) {
    // If we've got GPUs, assume we're using them
    if (!local_gpus.empty() || !local_omps.empty()) {
      ShardingFunction* function = find_sharding_functor(copy);
      const Processor proc =
          function->find_proc(copy.index_point, copy.index_domain);
      assert(
          (proc.kind() == Processor::OMP_PROC) ||
          (proc.kind() == Processor::TOC_PROC));
      if (proc.kind() == Processor::OMP_PROC)
        target_memory = local_numa_domains[proc];
      else
        target_memory = local_frame_buffers[proc];
    }
  } else {
    // If we have just one local GPU then let's use it, otherwise punt to CPU
    // since it's not clear which one we should use
    if (local_frame_buffers.size() == 1)
      target_memory = local_frame_buffers.begin()->second;
  }
  for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++) {
    const RegionRequirement& src_req = copy.src_requirements[idx];
    output.src_instances[idx].resize(src_req.privilege_fields.size());
    const std::vector<PhysicalInstance>& src_valid = input.src_instances[idx];
    unsigned fidx = 0;
    for (std::set<FieldID>::const_iterator it =
             src_req.privilege_fields.begin();
         it != src_req.privilege_fields.end(); it++) {
      if (find_existing_instance(
              src_req.region, *it, target_memory,
              output.src_instances[idx][fidx]) ||
          map_tensor(
              ctx, copy, idx, src_req.region, *it, target_memory,
              Processor::NO_PROC, src_valid, output.src_instances[idx][fidx]))
        needed_acquires.push_back(output.src_instances[idx][fidx]);
    }
    const RegionRequirement& dst_req = copy.dst_requirements[idx];
    output.dst_instances[idx].resize(dst_req.privilege_fields.size());
    const std::vector<PhysicalInstance>& dst_valid = input.dst_instances[idx];
    fidx = 0;
    for (std::set<FieldID>::const_iterator it =
             dst_req.privilege_fields.begin();
         it != dst_req.privilege_fields.end(); it++) {
      if (((dst_req.redop == 0) && find_existing_instance(
                                       dst_req.region, *it, target_memory,
                                       output.dst_instances[idx][fidx])) ||
          map_tensor(
              ctx, copy, copy.src_requirements.size() + idx, dst_req.region,
              *it, target_memory, Processor::NO_PROC, dst_valid,
              output.dst_instances[idx][fidx], dst_req.redop))
        needed_acquires.push_back(output.dst_instances[idx][fidx]);
    }
    if (idx < copy.src_indirect_requirements.size()) {
      const RegionRequirement& src_idx = copy.src_indirect_requirements[idx];
      assert(src_idx.privilege_fields.size() == 1);
      const FieldID fid = *(src_idx.privilege_fields.begin());
      const std::vector<PhysicalInstance>& idx_valid =
          input.src_indirect_instances[idx];
      if (find_existing_instance(
              src_idx.region, fid, target_memory,
              output.src_indirect_instances[idx]) ||
          map_tensor(
              ctx, copy, idx, src_idx.region, fid, target_memory,
              Processor::NO_PROC, idx_valid,
              output.src_indirect_instances[idx]))
        needed_acquires.push_back(output.src_indirect_instances[idx]);
    }
    if (idx < copy.dst_indirect_requirements.size()) {
      const RegionRequirement& dst_idx = copy.dst_indirect_requirements[idx];
      assert(dst_idx.privilege_fields.size() == 1);
      const FieldID fid = *(dst_idx.privilege_fields.begin());
      const std::vector<PhysicalInstance>& idx_valid =
          input.dst_indirect_instances[idx];
      if (find_existing_instance(
              dst_idx.region, fid, target_memory,
              output.dst_indirect_instances[idx]) ||
          map_tensor(
              ctx, copy, idx, dst_idx.region, fid, target_memory,
              Processor::NO_PROC, idx_valid,
              output.dst_indirect_instances[idx]))
        needed_acquires.push_back(output.dst_indirect_instances[idx]);
    }
  }
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(
             ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    // If we failed to acquire any of the instances we need to prune them
    // out of the mapper's data structure so do that first
    std::set<PhysicalInstance> failed_acquires;
    filter_failed_acquires(needed_acquires, failed_acquires);
    // Now go through and try to remap region requirements with failed
    // acquisitions
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++) {
      const RegionRequirement& src_req = copy.src_requirements[idx];
      const std::vector<PhysicalInstance>& src_valid = input.src_instances[idx];
      unsigned fidx = 0;
      for (std::set<FieldID>::const_iterator it =
               src_req.privilege_fields.begin();
           it != src_req.privilege_fields.end(); it++) {
        if (failed_acquires.find(output.src_instances[idx][fidx]) ==
            failed_acquires.end())
          continue;
        if (map_tensor(
                ctx, copy, idx, src_req.region, *it, target_memory,
                Processor::NO_PROC, src_valid, output.src_instances[idx][fidx]))
          needed_acquires.push_back(output.src_instances[idx][fidx]);
      }
      const RegionRequirement& dst_req = copy.dst_requirements[idx];
      output.dst_instances[idx].resize(dst_req.privilege_fields.size());
      const std::vector<PhysicalInstance>& dst_valid = input.dst_instances[idx];
      fidx = 0;
      for (std::set<FieldID>::const_iterator it =
               dst_req.privilege_fields.begin();
           it != dst_req.privilege_fields.end(); it++) {
        if (failed_acquires.find(output.dst_instances[idx][fidx]) ==
            failed_acquires.end())
          continue;
        if (map_tensor(
                ctx, copy, copy.src_requirements.size() + idx, dst_req.region,
                *it, target_memory, Processor::NO_PROC, dst_valid,
                output.dst_instances[idx][fidx], dst_req.redop))
          needed_acquires.push_back(output.dst_instances[idx][fidx]);
      }
      if (idx < copy.src_indirect_requirements.size()) {
        const RegionRequirement& src_idx = copy.src_indirect_requirements[idx];
        assert(src_idx.privilege_fields.size() == 1);
        const FieldID fid = *(src_idx.privilege_fields.begin());
        const std::vector<PhysicalInstance>& idx_valid =
            input.src_indirect_instances[idx];
        if ((failed_acquires.find(output.src_indirect_instances[idx]) !=
             failed_acquires.end()) &&
            map_tensor(
                ctx, copy, idx, src_idx.region, fid, target_memory,
                Processor::NO_PROC, idx_valid,
                output.src_indirect_instances[idx]))
          needed_acquires.push_back(output.src_indirect_instances[idx]);
      }
      if (idx < copy.dst_indirect_requirements.size()) {
        const RegionRequirement& dst_idx = copy.dst_indirect_requirements[idx];
        assert(dst_idx.privilege_fields.size() == 1);
        const FieldID fid = *(dst_idx.privilege_fields.begin());
        const std::vector<PhysicalInstance>& idx_valid =
            input.dst_indirect_instances[idx];
        if ((failed_acquires.find(output.dst_indirect_instances[idx]) !=
             failed_acquires.end()) &&
            map_tensor(
                ctx, copy, idx, dst_idx.region, fid, target_memory,
                Processor::NO_PROC, idx_valid,
                output.dst_indirect_instances[idx]))
          needed_acquires.push_back(output.dst_indirect_instances[idx]);
      }
    }
  }
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_copy_sources(
    const MapperContext ctx, const Copy& copy, const SelectCopySrcInput& input,
    SelectCopySrcOutput& output)
//--------------------------------------------------------------------------
{
  triton_select_sources(
      ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void
StrategyMapper::speculate(
    const MapperContext ctx, const Copy& copy, SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_profiling(
    const MapperContext ctx, const Copy& copy, const CopyProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling for copies yet
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const Copy& copy,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = find_sharding_functor(copy)->sharding_id;
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_close(
    const MapperContext ctx, const Close& close, const MapCloseInput& input,
    MapCloseOutput& output)
//--------------------------------------------------------------------------
{
  // Map everything with composite instances for now
  output.chosen_instances.push_back(PhysicalInstance::get_virtual_instance());
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_close_sources(
    const MapperContext ctx, const Close& close,
    const SelectCloseSrcInput& input, SelectCloseSrcOutput& output)
//--------------------------------------------------------------------------
{
  triton_select_sources(
      ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_profiling(
    const MapperContext ctx, const Close& close,
    const CloseProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling yet for triton
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const Close& close,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_acquire(
    const MapperContext ctx, const Acquire& acquire,
    const MapAcquireInput& input, MapAcquireOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do
}

//--------------------------------------------------------------------------
void
StrategyMapper::speculate(
    const MapperContext ctx, const Acquire& acquire, SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_profiling(
    const MapperContext ctx, const Acquire& acquire,
    const AcquireProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling for triton yet
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const Acquire& acquire,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_release(
    const MapperContext ctx, const Release& release,
    const MapReleaseInput& input, MapReleaseOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_release_sources(
    const MapperContext ctx, const Release& release,
    const SelectReleaseSrcInput& input, SelectReleaseSrcOutput& output)
//--------------------------------------------------------------------------
{
  triton_select_sources(
      ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void
StrategyMapper::speculate(
    const MapperContext ctx, const Release& release, SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_profiling(
    const MapperContext ctx, const Release& release,
    const ReleaseProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling for triton yet
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const Release& release,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_partition_projection(
    const MapperContext ctx, const Partition& partition,
    const SelectPartitionProjectionInput& input,
    SelectPartitionProjectionOutput& output)
//--------------------------------------------------------------------------
{
  // If we have an open complete partition then use it
  if (!input.open_complete_partitions.empty())
    output.chosen_partition = input.open_complete_partitions[0];
  else
    output.chosen_partition = LogicalPartition::NO_PART;
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_partition(
    const MapperContext ctx, const Partition& partition,
    const MapPartitionInput& input, MapPartitionOutput& output)
//--------------------------------------------------------------------------
{
  const RegionRequirement& req = partition.requirement;
  output.chosen_instances.resize(req.privilege_fields.size());
  const std::vector<PhysicalInstance>& valid = input.valid_instances;
  std::vector<PhysicalInstance> needed_acquires;
  unsigned fidx = 0;
  for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
       it != req.privilege_fields.end(); it++) {
    if (find_existing_instance(
            req.region, *it, local_system_memory,
            output.chosen_instances[fidx]) ||
        map_tensor(
            ctx, partition, 0, req.region, *it, local_system_memory,
            Processor::NO_PROC, valid, output.chosen_instances[fidx])) {
      needed_acquires.push_back(output.chosen_instances[fidx]);
    }
  }
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(
             ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    std::set<PhysicalInstance> failed_instances;
    filter_failed_acquires(needed_acquires, failed_instances);
    // Now go through all the fields for the instances and try and remap
    std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
    for (unsigned idx = 0; idx < output.chosen_instances.size(); idx++, fit++) {
      if (failed_instances.find(output.chosen_instances[idx]) ==
          failed_instances.end())
        continue;
      // Now try to remap it
      if (map_tensor(
              ctx, partition, 0 /*idx*/, req.region, *fit, local_system_memory,
              Processor::NO_PROC, valid, output.chosen_instances[idx]))
        needed_acquires.push_back(output.chosen_instances[idx]);
    }
  }
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_partition_sources(
    const MapperContext ctx, const Partition& partition,
    const SelectPartitionSrcInput& input, SelectPartitionSrcOutput& output)
//--------------------------------------------------------------------------
{
  triton_select_sources(
      ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void
StrategyMapper::report_profiling(
    const MapperContext ctx, const Partition& partition,
    const PartitionProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling yet
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const Partition& partition,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = find_sharding_functor(partition)->sharding_id;
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const Fill& fill,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = find_sharding_functor(fill)->sharding_id;
}

//--------------------------------------------------------------------------
void
StrategyMapper::configure_context(
    const MapperContext ctx, const Task& task, ContextConfigOutput& output)
//--------------------------------------------------------------------------
{
  // Use the defaults currently
}

//--------------------------------------------------------------------------
void
StrategyMapper::pack_tunable(
    const int value, Mapper::SelectTunableOutput& output)
//--------------------------------------------------------------------------
{
  int* result = (int*)malloc(sizeof(value));
  *result = value;
  output.value = result;
  output.size = sizeof(value);
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_tunable_value(
    const MapperContext ctx, const Task& task, const SelectTunableInput& input,
    SelectTunableOutput& output)
//--------------------------------------------------------------------------
{
  // No tunable values at the moment
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_sharding_functor(
    const MapperContext ctx, const MustEpoch& epoch,
    const SelectShardingFunctorInput& input,
    MustEpochShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  // No must epoch launches in trition
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::memoize_operation(
    const MapperContext ctx, const Mappable& mappable,
    const MemoizeInput& input, MemoizeOutput& output)
//--------------------------------------------------------------------------
{
  output.memoize = true;
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_must_epoch(
    const MapperContext ctx, const MapMustEpochInput& input,
    MapMustEpochOutput& output)
//--------------------------------------------------------------------------
{
  // No must epoch launches in triton
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::map_dataflow_graph(
    const MapperContext ctx, const MapDataflowGraphInput& input,
    MapDataflowGraphOutput& output)
//--------------------------------------------------------------------------
{
  // Not supported yet
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_tasks_to_map(
    const MapperContext ctx, const SelectMappingInput& input,
    SelectMappingOutput& output)
//--------------------------------------------------------------------------
{
  // Just map all the ready tasks
  for (std::list<const Task*>::const_iterator it = input.ready_tasks.begin();
       it != input.ready_tasks.end(); it++)
    output.map_tasks.insert(*it);
}

//--------------------------------------------------------------------------
void
StrategyMapper::select_steal_targets(
    const MapperContext ctx, const SelectStealingInput& input,
    SelectStealingOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do, no stealing in the leagte mapper currently
}

//--------------------------------------------------------------------------
void
StrategyMapper::permit_steal_request(
    const MapperContext ctx, const StealRequestInput& input,
    StealRequestOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do, no stealing in the triton mapper currently
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::handle_message(
    const MapperContext ctx, const MapperMessage& message)
//--------------------------------------------------------------------------
{
  // We shouldn't be receiving any messages currently
  abort();
}

//--------------------------------------------------------------------------
void
StrategyMapper::handle_task_result(
    const MapperContext ctx, const MapperTaskResult& result)
//--------------------------------------------------------------------------
{
  // Nothing to do since we should never get one of these
  abort();
}

}}}  // namespace triton::backend::legion

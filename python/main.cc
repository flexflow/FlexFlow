/* Copyright 2020 Stanford University, Los Alamos National Laboratory
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

#include "legion.h"
#include "null_mapper.h"
#include "realm/python/python_module.h"
#include "realm/python/python_source.h"

#include <libgen.h>

using namespace Legion;
using namespace Legion::Mapping;

void register_flexflow(int argc, char **argv);

static bool control_replicate = true;
static const char * const unique_name = "legion_python";
static const VariantID vid = 1;

class LegionPyShardingFunctor : public ShardingFunctor {
public:
  LegionPyShardingFunctor(void) { }
  virtual ~LegionPyShardingFunctor(void) { }
public:
  virtual ShardID shard(const DomainPoint &point,
                        const Domain &full_space,
                        const size_t total_shards);
};

// Special mapper just for mapping the top-level Python tasks
class LegionPyMapper : public Legion::Mapping::NullMapper {
public:
  LegionPyMapper(MapperRuntime *runtime, Machine machine, 
                 TaskID top_task_id, ShardingID sharding_id);
  virtual ~LegionPyMapper(void);
public:
  static AddressSpaceID get_local_node(void);
  static size_t get_total_nodes(Machine m);
  static const char* create_name(AddressSpace node);
public:
  virtual const char* get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;
  virtual bool request_valid_instances(void) const { return false; }
public: // Task mapping calls
  virtual void select_task_options(const MapperContext    ctx,
                                   const Task&            task,
                                         TaskOptions&     output);
  virtual void slice_task(const MapperContext      ctx,
                          const Task&              task, 
                          const SliceTaskInput&    input,
                                SliceTaskOutput&   output);
  virtual void map_task(const MapperContext      ctx,
                        const Task&              task,
                        const MapTaskInput&      input,
                              MapTaskOutput&     output);
  virtual void map_replicate_task(const MapperContext      ctx,
                                  const Task&              task,
                                  const MapTaskInput&      input,
                                  const MapTaskOutput&     default_output,
                                  MapReplicateTaskOutput&  output);
  virtual void select_sharding_functor(
                             const MapperContext                ctx,
                             const Task&                        task,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output);
  virtual void select_tunable_value(const MapperContext         ctx,
                                    const Task&                 task,
                                    const SelectTunableInput&   input,
                                          SelectTunableOutput&  output);
  virtual void select_steal_targets(const MapperContext         ctx,
                                    const SelectStealingInput&  input,
                                          SelectStealingOutput& output);
  virtual void select_tasks_to_map(const MapperContext          ctx,
                                   const SelectMappingInput&    input,
                                         SelectMappingOutput&   output);
public:
  virtual void configure_context(const MapperContext         ctx,
                                 const Task&                 task,
                                       ContextConfigOutput&  output);
protected:
  void map_top_level_task(const MapperContext ctx,
                          const Task& task,
                          const MapTaskInput& input,
                                MapTaskOutput& output);
public:
  const AddressSpace local_node;
  const size_t total_nodes;
  const char *const mapper_name;
  const TaskID top_task_id;
  const ShardingID sharding_id;
protected:
  std::vector<Processor> local_pys; // Python processors
};

static void python_main_callback(Machine machine, Runtime *runtime,
                                 const std::set<Processor> &local_procs)
{
  // Get an ID for the top-level task, register it with the runtime
  const TaskID top_task_id = runtime->generate_library_task_ids(unique_name, 3); 
  runtime->set_top_level_task_id(top_task_id);
  // Register a variant for the top-level task
  {
    const char *const task_name = "legion_python_main";
    TaskVariantRegistrar registrar(top_task_id, task_name, false/*global*/);
    registrar.add_constraint(ProcessorConstraint(Processor::PY_PROC));
    CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
    code_desc.add_implementation(
        new Realm::PythonSourceImplementation("legion_top", task_name));
    registrar.set_replicable(control_replicate);
    runtime->register_task_variant(registrar, code_desc, NULL, 0, 0/*no return*/, vid);
    runtime->attach_name(top_task_id, task_name, false/*mutable*/, true/*local only*/);
  }
  // Register a variant for the global import task
  {
    const char *const task_name = "legion_python_import_global";
    TaskVariantRegistrar registrar(top_task_id+1, task_name, false/*global*/);
    registrar.add_constraint(ProcessorConstraint(Processor::PY_PROC));
    CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
    code_desc.add_implementation(
        new Realm::PythonSourceImplementation("legion_top", task_name));
    runtime->register_task_variant(registrar, code_desc, NULL, 0, 0/*no return*/, vid);
    runtime->attach_name(top_task_id+1, task_name, false/*mutable*/, true/*local only*/);
  }
  // Register a variant for the cleanup task
  {
    const char *const task_name = "legion_python_cleanup";
    TaskVariantRegistrar registrar(top_task_id+2, task_name, false/*global*/);
    registrar.add_constraint(ProcessorConstraint(Processor::PY_PROC));
    CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
    code_desc.add_implementation(
        new Realm::PythonSourceImplementation("legion_top", task_name));
    runtime->register_task_variant(registrar, code_desc, NULL, 0, 0/*no return*/, vid);
    runtime->attach_name(top_task_id+2, task_name, false/*mutable*/, true/*local only*/);
  }
  // Register our sharding function for any global import tasks
  const ShardingID sharding_id = runtime->generate_library_sharding_ids(unique_name, 1);
  runtime->register_sharding_functor(sharding_id, new LegionPyShardingFunctor());
  // Register our mapper for the top-level task
  const MapperID top_mapper_id = runtime->generate_library_mapper_ids(unique_name, 1);
  runtime->set_top_level_task_mapper_id(top_mapper_id);
  runtime->add_mapper(top_mapper_id,
      new LegionPyMapper(runtime->get_mapper_runtime(), machine, top_task_id, sharding_id));
}

int main(int argc, char **argv)
{
  // Make sure argc and argv are valid before we look at them
  Runtime::initialize(&argc, &argv, false/*filter legion and realm args*/);
#ifdef BINDINGS_AUGMENT_PYTHONPATH
  // Add the binary directory to PYTHONPATH. This is needed for
  // in-place builds to find legion.py.

  // Do this before any threads are spawned.
  {
    char *bin_path = strdup(argv[0]);
    assert(bin_path != NULL);
    char *bin_dir = dirname(bin_path);

    char *previous_python_path = getenv("PYTHONPATH");
    if (previous_python_path != 0) {
      size_t bufsize = strlen(previous_python_path) + strlen(bin_dir) + 2;
      char *buffer = (char *)calloc(bufsize, sizeof(char));
      assert(buffer != 0);

      // assert(strlen(previous_python_path) + strlen(bin_dir) + 2 < bufsize);
      // Concatenate bin_dir to the end of PYTHONPATH.
      bufsize--;
      strncat(buffer, previous_python_path, bufsize);
      bufsize -= strlen(previous_python_path);
      strncat(buffer, ":", bufsize);
      bufsize -= strlen(":");
      strncat(buffer, bin_dir, bufsize);
      bufsize -= strlen(bin_dir);
      setenv("PYTHONPATH", buffer, true /*overwrite*/);
    } else {
      setenv("PYTHONPATH", bin_dir, true /*overwrite*/);
    }

    free(bin_path);
  }
#endif

#ifdef BINDINGS_DEFAULT_MODULE
#define str(x) #x
  Realm::Python::PythonModule::import_python_module(str(BINDINGS_DEFAULT_MODULE));
#undef str
#endif

  int start = 1;
  if ((argc > start) && (strcmp(argv[start],"--nocr") == 0)) {
    control_replicate = false;
    start++;
  }
  if ((argc > start) && argv[start][0] == '-') {
    if (argv[start][1] == '\0')
      // Interactive console means no control replication
      control_replicate = false;
  // Note this check is safe because we filtered all the 
  // Legion and Realm flags out earlier
  } else if (argc < (start+1)) {
    // Ineractive console means no control replication
    control_replicate = false;
  }

  register_flexflow(argc, argv);

  Runtime::add_registration_callback(python_main_callback);

  return Runtime::start(argc, argv);
}

LegionPyMapper::LegionPyMapper(MapperRuntime *rt, Machine m, TaskID top_id, ShardingID sid) 
  : NullMapper(rt, m), local_node(get_local_node()), 
    total_nodes(get_total_nodes(m)), mapper_name(create_name(local_node)),
    top_task_id(top_id), sharding_id(sid)
{
  Machine::ProcessorQuery py_procs(machine);
  py_procs.local_address_space();
  py_procs.only_kind(Processor::PY_PROC);
  for (Machine::ProcessorQuery::iterator it = 
        py_procs.begin(); it != py_procs.end(); it++)
    local_pys.push_back(*it);
  if (local_pys.empty())
  {
    fprintf(stderr,"FATAL: Legion Python found no Python processors!\n");
    fprintf(stderr,"Please run with at least '-ll:py 1' on the command line.\n");
    exit(1);
  }
}

LegionPyMapper::~LegionPyMapper(void)
{
  free(const_cast<char*>(mapper_name));
}

/*static*/ AddressSpace LegionPyMapper::get_local_node(void)
{
  Processor p = Processor::get_executing_processor();
  return p.address_space();
}

/*static*/ size_t LegionPyMapper::get_total_nodes(Machine m)
{
  Machine::ProcessorQuery query(m);
  query.only_kind(Processor::PY_PROC);
  std::set<AddressSpace> spaces;
  for (Machine::ProcessorQuery::iterator it = query.begin(); 
        it != query.end(); it++)
    spaces.insert(it->address_space());
  return spaces.size();
}

/*static*/ const char* LegionPyMapper::create_name(AddressSpace node)
{
  char buffer[128];
  snprintf(buffer, 127, "Legion Python Mapper on Node %d", node);
  return strdup(buffer);
}

const char* LegionPyMapper::get_mapper_name(void) const    
{
  return mapper_name;
}

Mapper::MapperSyncModel LegionPyMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void LegionPyMapper::select_task_options(const MapperContext    ctx,
                                         const Task&            task,
                                               TaskOptions&     output)
{
  if (task.task_id == top_task_id)
  {
    assert(task.get_depth() == 0);
    // We only control replicate if we're allowed to and there are multiple nodes
    output.replicate = control_replicate && (total_nodes > 1);
  }
  else
  {
    assert(task.task_id <= (top_task_id + 2));
    output.replicate = false;
  }
  assert(!local_pys.empty());
  output.initial_proc = local_pys.front();
}

void LegionPyMapper::slice_task(const MapperContext      ctx,
                                const Task&              task, 
                                const SliceTaskInput&    input,
                                      SliceTaskOutput&   output)
{
  assert(task.task_id <= (top_task_id + 2));
  const Rect<1> bounds = input.domain; 
  const size_t num_points = bounds.volume();
  output.slices.reserve(num_points);
  if (num_points == local_pys.size())
  {
    unsigned index = 0;
    // Already been sharded, just assign to the local python procs
    for (coord_t p = bounds.lo[0]; p <= bounds.hi[0]; p++)
    {
      const Point<1> point(p);
      const Rect<1> rect(point,point);
      output.slices.push_back(TaskSlice(Domain(rect),
            local_pys[index++], false/*recurse*/, false/*stelable*/));
    }
  }
  else
  {
    // Not sharded, so we should have points for all the python procs
    assert(input.domain.get_volume() == (local_pys.size() * total_nodes));
    Machine::ProcessorQuery py_procs(machine);
    py_procs.only_kind(Processor::PY_PROC);
    std::set<AddressSpaceID> spaces;
    for (Machine::ProcessorQuery::iterator it = 
          py_procs.begin(); it != py_procs.end(); it++)
    {
      const AddressSpaceID space = it->address_space();
      if (spaces.find(space) != spaces.end())
        continue;
      const Point<1> lo(space*local_pys.size());
      const Point<1> hi((space+1)*local_pys.size()-1);
      const Rect<1> rect(lo,hi);
      output.slices.push_back(TaskSlice(Domain(rect),
            *it, true/*recurse*/, false/*stelable*/));
      spaces.insert(space);
    }
  }
}

void LegionPyMapper::map_task(const MapperContext      ctx,
                              const Task&              task,
                              const MapTaskInput&      input,
                                    MapTaskOutput&     output)
{
  if (task.task_id == top_task_id)
  {
    assert(task.get_depth() == 0);
    map_top_level_task(ctx, task, input, output);
  }
  else
  {
    assert(task.task_id <= (top_task_id + 2));
    assert(task.regions.empty());
    output.chosen_variant = vid;
  }
  // Still need to fill in the target procs
  assert(task.target_proc.kind() == Processor::PY_PROC);
  output.target_procs.push_back(task.target_proc);
}

void LegionPyMapper::map_replicate_task(const MapperContext      ctx,
                                        const Task&              task,
                                        const MapTaskInput&      input,
                                        const MapTaskOutput&     def_output,
                                        MapReplicateTaskOutput&  output)
{
  assert(task.get_depth() == 0);
  MapTaskOutput top_level_mapping = def_output;
  map_top_level_task(ctx, task, input, top_level_mapping);
  assert(output.task_mappings.empty()); // need the resize to write
  output.task_mappings.resize(total_nodes, top_level_mapping);
  output.control_replication_map.resize(total_nodes);
  // Now fill in the set of processors
  Machine::ProcessorQuery py_procs(machine);
  py_procs.only_kind(Processor::PY_PROC);
  std::set<AddressSpace> handled;
  for (Machine::ProcessorQuery::iterator it = py_procs.begin();
        it != py_procs.end(); it++)
  {
    const AddressSpace space = it->address_space();
    // See if we've already seen it
    if (handled.find(space) != handled.end())
      continue;
    output.task_mappings[space].target_procs.push_back(*it);
    output.control_replication_map[space] = *it;
    handled.insert(space);
  }
}

void LegionPyMapper::map_top_level_task(const MapperContext ctx,
                                        const Task& task,
                                        const MapTaskInput& input,
                                              MapTaskOutput& output)
{
  assert(task.get_depth() == 0);
  assert(task.regions.empty());
  output.chosen_variant = vid;
}

ShardID LegionPyShardingFunctor::shard(const DomainPoint &point,
                                       const Domain &full_domain,
                                       size_t total_shards)
{
  Point<1> p = point;
  Rect<1> bounds = full_domain;
  const size_t volume = bounds.volume();
  assert((volume % total_shards) == 0);
  const size_t pernode = volume / total_shards;
  return (p[0] / pernode);
}

void LegionPyMapper::select_sharding_functor(
                             const MapperContext                ctx,
                             const Task&                        task,
                             const SelectShardingFunctorInput&  input,
                                   SelectShardingFunctorOutput& output)
{
  assert(task.task_id == (top_task_id+1));
  output.chosen_functor = sharding_id;
}

void LegionPyMapper::select_tunable_value(const MapperContext         ctx,
                                          const Task&                 task,
                                          const SelectTunableInput&   input,
                                                SelectTunableOutput&  output)
{
  // We only have one tunable value for now and its the global number of python procs
  assert(input.tunable_id == 0);
  const int value = local_pys.size() * total_nodes;
  int *result = (int*)malloc(sizeof(value));
  *result = value;
  output.value = result;
  output.size = sizeof(value);
}

void LegionPyMapper::select_steal_targets(const MapperContext         ctx,
                                          const SelectStealingInput&  input,
                                                SelectStealingOutput& output)
{
  // Do nothing
}

void LegionPyMapper::select_tasks_to_map(const MapperContext          ctx,
                                         const SelectMappingInput&    input,
                                               SelectMappingOutput&   output)
{
  output.map_tasks.insert(input.ready_tasks.begin(), input.ready_tasks.end());
}

void LegionPyMapper::configure_context(const MapperContext         ctx,
                                       const Task&                 task,
                                             ContextConfigOutput&  output)
{
  // Use the defaults currently 
}


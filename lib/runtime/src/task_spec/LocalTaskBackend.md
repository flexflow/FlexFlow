# `LocalTaskBackend` Design

## Allocator

The lowest level interface is the memory `Allocator`. This will allocate and de-allocate memory as follows:
```
class Allocator {
  Allocator(total_size);
  
  int total_size;
  int total_used;

  allocate(ptr, size) {
    if (total_used + size < total_size) {
      cudaMalloc(ptr, size);
      total_used += size;
    }
  }

  deallocate(ptr, size) {
    cudaFree(ptr);
    total_size -= size;
  }
}
```

## Retriever

Next, the `Retriever` allows the backend to fetch the memory region or argument associated with a task invocation.

```
class Retriever {
  map<task_id, map<slot_id, tensor_type>> task_tensor_mapping;
  map<task_id, map<slot_id, arg_type>> task_arg_mapping;

  tensor_type get_tensor(task_id, slot_id) {
    return task_tensor_mapping[task_id][slot_id];
  }
  arg_type get_arg(...) {}

  set_tensor(task_id, slot_id, tensor_type) {
    task_tensor_mapping[task_id][slot_id] = tensor_type;
  }
  set_arg(...) {}

  set_tensor_dependency(task_id, slot_id, source_task_id, source_slot_id) {
    task_tensor_mapping[task_id][slot_id] = task_tensor_mapping[source_task_id][source_slot_id];
    ref_count[source_task_id][source_slot_id]++;
  }
  set_arg_dependency(...) {}
}
```

## Task Graph

The `TaskGraph` maintains the order of task execution and the dependencies between task slots.

```
class TaskGraph {
  class TaskNode {
    task_id;
    map<slot_id, (task_id, slot_id)> tensor_dependency_map;
    map<slot_id, (task_id, slot_id)> arg_dependency_map;
    map<slot_id, size> tensor_slots;
    set<slot_id, arg_type> arg_slots;
  }

  vector<TaskNodes> nodes;
}
```

## Backend

```
class Backend {
  Allocator a;
  Retriever r;

  insert_graph(TaskGraph g) {
    for (task in g.nodes) {
      insert_task(task)
    }
  }

  insert_task(TaskNode t) {
    for (slot_id, size in t.tensor_slots) {
      tensor_slot_ptr;
      a.allocate(tensor_slot_ptr, size);
      r.set_tensor(t.task_id, slot_id, tensor_slot_ptr);
    }

    for (slot_id in t.arg_slots) {
      r.set_arg(t.task_id, slot_id, arg_type);
    }

    for (slot_id, source_task in t.tensor_dependency_map) {
      r.set_tensor_dependency(t.task_id, slot_id, source_task.task_id, source_task.slot_id);
    }

    for (slot_id, source_task in t.arg_dependency_map) {
      r.set_arg_dependency(t.task_id, slot_id, source_task.task_id, source_task.slot_id);
    }
  }

  tensor_type get_tensor(task_id, slot_id) {
    return r.get_tensor(task_id, slot_id);
  }
  
  arg_type get_arg(task_id, slot_id) {
    return r.get_arg(task_id, slot_id);
  }

  // TODO: fuse task, launch task, and cleanup task

}
```


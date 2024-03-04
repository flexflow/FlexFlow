# Local Backend Internals

A step-by-step walkthrough.

## Compilation

During compilation, a `ComputationGraph` is generated from a `torch.fxGraphModule`. Since we have no need to run a PCG, we can skip the optimization step. Instead, the `ComputationGraph` itself can be considered as the `CompiledModel` and fed to the serializer. Similarly, we can strip down the legion/PCG from `ModelTrainingInstance` and initialize a `LocalModelTrainingInstance` as follows:
```
struct LocalModelTrainingInstance {
  ComputationGraph computation_graph;
  EnableProfiling enable_profiling;
  LocalBackend backend;

  void initialize_backend(size_t mem_size) {
    backend = LocalBackend(mem_size);
  }

  GenericTensorAccessorR forward() {
    return backend.forward(computation_graph);
  }

  GenericTensorAccessorR backward() {
    return backend.backward(computation_graph);
  }

  void update() {
    backend.update(computation_graph);
  }
};
```

Instead of returning `TypedFuture<>` for `forward()` and `backward()`, we can just return the tensor itself since local backend operates synchronously.

## Runtime

In the training loop, `LocalModelTrainingInstance::forward` will invoke `LocalBackend::forward` on its computation graph.
```
struct LocalBackend {
  Allocator allocator;

  map<task_id_t, vector<std::type_index> arg_store;
  map<task_id_t, vector<GenericTensorAccessorW *>> tensor_store;

  map<pair<task_id_t, int>, pair<task_id_t, int>> arg_dependency_map;
  map<pair<task_id_t, int>, pair<task_id_t, int>> tensor_dependency_map;

  LocalBackend(size_t mem_size) { 
    allocator = Allocator(mem_size); 
  }

  F register_task(task_id_t, OpTaskSignature sig, F task_impl) {
    for (auto arg: sig.get_arg_slots()) {
      if (arg_dependency_map[pair<task_id_t, i>]) {
        arg_store[task_id_t].push_back(...) // add argument from other task
      } else {
        arg_store[task_id_t].push_back(arg);
      }
    }
    
    for (auto tensor: sig.get_tensor_slots()) {
      if (tensor_dependency_map[pair<task_id_t, i>]) {
        tensor_store[task_id_t].push_back(...) // add tensor from other task
      } else {
        GenericTensorAccessorW tensor_acc = allocator.allocate(tensor);
        tensor_store[task_id_t].push_back(&tensor_acc);
      }
    } 

    return task_impl;
  }

  TaskArgumentAccessor get_args(OpTaskInvocation const & inv) {
    // TODO -- package args and tensors for a particular task
  }

  GenericTensorAccessorR forward(ComputationGraph const & cg) {
    set_dependency_maps(cg);

    GenericTensorAccessorR output_tensor;

    for (auto op: top_sort(cg)) {
      F op_task_impl = register_task<get_fwd_task_id(op)>();
      OpTaskInvocation invocation = forward(get_operator_attrs(op));
      TaskArgumentAccessor acc = get_args(invocation);
      output_tensor = op_task_impl(acc);
    }

    return output_tensor;
  }

}
```

Similar process for `backward()` and `update()`.
# Local Backend Internals

A step-by-step walkthrough.

## Compilation

During compilation, a `ComputationGraph` is generated. We can use this to initialize a `RuntimeBacking` (`LocalTaskBackend` is-a `RuntimeBacking`).

```
LocalTaskBackend(ComputationGraph cg) {
  for (auto op: top_sort(cg)) {
    register_task<get_fwd_task_id(op)>();
  }

  for (auto op: reverse(top_sort(cg))) {
    register_task<get_bwd_task_id(op)>();
  }

  register_task<UPDATE_TASK_ID>();
}

struct TaskStore {
  map<task_id, vector<args>> arg_store;
  map<task_id, vector<tensor>> tensor_store;

  add_arg(id, arg) {
    arg_store[id].push_back(arg);
  }

  add_tensor(id, tensor) {
    tensor_store[id].push_back(tensor);
  }
}

void register_task(task_id_t, TaskSignature sig, F task_impl) {
  task_impl_map[task_id_t] = task_impl;

  for (auto arg: sig.get_arg_slots()) {
    task_store.add_arg(task_id_t, arg)
  }
  
  for (auto tensor: sig.get_tensor_slots()) {
    task_store.add_tensor(task_id_t, tensor);
  } 
}
```

## Runtime

At runtime, we call `forward()`, `backward()`, and `update()` and pass in a `ModelTrainingInstance` which holds references to the `ComputationGraph` and the `RuntimeBacking`.

```
void forward(ModelTrainingInstance m) {
  for (auto op: top_sort(m.cg)) {
    OpTaskInvocation invocation = forward(get_operator_attrs(op));
    execute(invocation);
  }
}

void backward(ModelTrainingInstance m) {
  for (auto op: reverse(top_sort(m.cg))) {
    OpTaskInvocation invocation = backward(get_operator_attrs(op));
    execute(invocation);
  }
}

void update(ModelTrainingInstance m) {
  TaskInvocation invocation = {...};
  execute(invocation);
}
```

`execute()` will invoke the backend, which will execute the task itself.

```
execute(OpTaskInvocation) {
  TaskArgumentAccessor acc = backend.get_args(invocation);
  F op_task_impl = backend.get_task_impl(invocation);
  op_task_impl(acc);
}
```

The backend functions.

```
TaskArgumentAccessor get_args(OpTaskInvocation) {
  return TaskArgumentAccessor(task_store.get_task_args(invocation.task_id));
}

F get_task_impl(OpTaskInvocation) {
  return task_impl_map[invocation.task_id];
}
```
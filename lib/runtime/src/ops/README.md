# `Operators`

This document describes the general outline that an operator developer needs to take care of when implementing a new operator.


## Slots

The Operator takes tensor inputs and outputs other tensors. The input tensors’ attributes and type need to be specified in the form of Slots, which has traits distinguished by an enumerator.

ex. attention.cc:
```cpp
enum Slots {
  QUERY_PARALLEL_TENSOR_SHAPE,
  KEY_PARALLEL_TENSOR_SHAPE,
  VALUE_PARALLEL_TENSOR_SHAPE,
  QPROJSIZE,
  KPROJSIZE,
  VPROJSIZE,
  OPROJSIZE,
  ATTRS,
  PROFILING,
  QUERY,
  KEY,
  VALUE,
  WEIGHTS,
  OUTPUT,
  HANDLE,
  PER_DEVICE_STATE
};

```

## Register Task

For the backing to use the operator, the operator needs to register its tasks with the backing. Generally, three tasks need to be registered, using register_task. 

The three tasks are as follows: 

<NAME_OF_OPERATOR>_INIT_TASK
<NAME_OF_OPERATOR>_FWD_TASK
<NAME_OF_OPERATOR>_BWD_TASK

ex. attention.cc
```cpp
void register_task<ATTENTION_INIT_TASK_ID>() 
void register_task<ATTENTION_FWD_TASK_ID>()
void register_task<ATTENTION_BWD_TASK_ID>()

```

Each of these register task methods registers the corresponding task function, along with its ID, name, and signature.

ex. attention.cc
```cpp
 register_task(ATTENTION_FWD_TASK_ID,
                "Attention Fwd",
                fwd_signature<ATTENTION_FWD_TASK_ID>(),
                forward_task);
}

```

## Signature

The function signature (fwd_signature, bwd_signature, init_signature) constructs a task signature instance and fills in all the parameters and its types (specified by slots) and returns the task signature.

ex. attention.cc
```cpp
 template <>
OpTaskSignature fwd_signature<ATTENTION_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);
 
 # parameters required for fwd_task are QUERY, KEY, VALUE, WEIGHTS, OUTPUTS
 # their types are input, output or weight tensors
  fwd.add_input_slot(QUERY);
  fwd.add_input_slot(KEY);
  fwd.add_input_slot(VALUE);
  fwd.add_weight_slot(WEIGHTS);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  # some parameters are checked, some are unchecked
  fwd.add_unchecked_arg_slot<MHAPerDeviceState>(PER_DEVICE_STATE);

  return fwd;
}


```

## bind vs. bind_arg

A quick note on the difference between bind vs. bind_arg: The bind method is used to associate a specific task or computation with an initial tensor. On the other hand, bind_arg has the same use, but for non-tensor values.

## Invocations

When the backing invokes any task, registered above, along with its parameters, its corresponding impl function is called. This follows the interface-implementation
design pattern

ex. attention.cc
```cpp
 static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {

  # accessor contains all the required parameters
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

}


```

Inside the implementation function (impl), all the parameters are extracted from the accessor, and the operator kernel is launched, along with all of its required tensors.

ex. attention.cc inside fwd_task_impl
```cpp
 # using accessor to extract the parameters needed for the kernel
  auto query = acc.get_tensor<Permissions::RO>(QUERY);
  auto key = acc.get_tensor<Permissions::RO>(KEY);
  auto value = acc.get_tensor<Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

```

later inside the impl function...

```cpp
 # Launching forward kernel
return profile(forward_kernel,
                 profiling,
                 "[MultiHeadAttention] forward_time = %.2lfms\n",
                 per_device_state,
                 query.get_float_ptr(),
                 key.get_float_ptr(),
                 value.get_float_ptr(),
                 weight.get_float_ptr(),
                 output.get_float_ptr());
}


```

## Operator Cost (metrics):

Every operator has a cost function, called measure_operator_cost. This method must invoke forward and backward implementation functions and collect time and memory profile data, using which metrics calculations are done and returned.

```cpp
float forward_time = forward_task_impl(fwd_accessor).value();
float backward_time = backward_task_impl(bwd_accessor).value();

float sync_time = default_estimate_sync_time(env);
return make_metrics(forward_time, backward_time, sync_time, env);

```

## X_INIT_TASK

This is a wrapper, which bundles all the tensor inputs, weights, and outputs & initializes the init kernel. It returns the device specific state.

```cpp
#specifies the input, weights, and outputs

 MultiHeadAttentionInputs<ParallelTensorShape> inputs =
      MultiHeadAttentionInputs<ParallelTensorShape>(
          query_parallel_tensor_shape,
          key_parallel_tensor_shape,
          value_parallel_tensor_shape);
  ParallelTensorShape output_parallel_tensor_shape =
      get_output_shape(attrs, inputs);
  ParallelTensorShape weight_parallel_tensor_shape =
      get_weights_shape(attrs, inputs);

```

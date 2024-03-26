
This document describes the general outline that an operator developer needs to take care of when implementing a new operator. Say we need to implement a new operator, specified by the signature below:

![](https://lh7-us.googleusercontent.com/Ep_J1q_2htSltLNw43LAFNi02YiG_ihwcmn6IZE1_uQBWWPliZKAwkSEPajQ-pR8wczDeenLyNY-iOwKZ89SJH2COoSFsHtpbK2Zm8Si1GTs6Q_9F6qhnoWccWBB6Wz3yaKfALmyT74i9vt_CuPmGYg)

To be able to implement such an operator in the FF environment, one should follow the steps indicated and explained/highlighted below:

  

1) Indicate each of the attributes/parameters (parameters a,b) as a Slot

2) f(a, b) is implemented in terms of 3 tasks invocations: init, forward, backward

3) Implement the task impl (implementation) functions for the init, forward, and backward task and their corresponding wrapper functions (return a + b)

4) Register each of the tasks using their corresponding task id, and provide task signature functions (defining the function)

5) Finally, provide a profiling calculations method

---------------------------------------------------------------------------------------------------------------------

## SLOTS:

One can think of the slots as the parameters a & b inside the def(a, b) line of our example. Slots are meaningful attributes for our operator, such as input tensors, weight tensors, and output tensors.

  

Underneath the surface, our backing Legion (which is like a compiler for our code), requires attributes to be in a certain order. We need the caller and callee to agree on the order of arguments, so by creating a layer of abstraction over directly corresponding indices and inputs, we now have argument names that match our input to the correct spot when making task invocations. You should trust that Legion understands these names and can decode what indices the names represent.

  

Simply put, the argument name is associated with a certain SLOT/index that the backing (Legion) understands. This is why we use an enum, so we ensure that every argument is associated with a unique number. It’s like the argument represents a code that Legion can decode to figure out what index it must put the argument’s value at.

  

An example of a simple set of Slots can be seen here:

```cpp
enum Slots {
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
  

Arguments are operator specific, but in the usual case, your operator will have input tensor(s), output tensor(s), and/or weight tensor(s). All operators share an argument for profiling, attributes (attrs), and many of them have some per-device state. The exact meaning of these attributes will be explored in the following sections. Just know that, QUERY, KEY, & VALUE are input tensors, we have output and weight tensors, and ATTRS and PROFILING are necessary.

  

**A quick note on ATTRS, PROFILING, and PER_DEVICE_STATE:**

Technically they're just enums, but in practice ATTRS usually denotes the compiler's representation of the operator in the CG/PCG, PROFILING is a boolean flag denoting whether or not the operator's kernel should be run with profiling/task metrics enabled, and HANDLE is a set of system-wide handles/references for things like nccl, cublas, cudnn, etc. PER_DEVICE_STATE refers to all the relevant metadata needed for the forward and backward pass

  

---------------------------------------------------------------------------------------------------------------------

  

## f(x, y) is implemented in terms of 3 tasks invocations: init, forward, backward

  

**Important distinctions:**
Task Invocations represents having the call f(x, y), but it’s not yet using the Legion backing to compile the statement. The task invocation is also very important because it tells us where we’re getting our attributes (x, y) from & what type we can expect for those.

  

The developer can specify the type of each operator attribute in the following methods: init, forward, and backward. Every task is associated with an OpTaskBinding and the above function, we associate the task id with its binding.

  

**bind vs. bind_arg:**
A quick note on the difference between bind vs. bind_arg: The bind method is used to create a placeholder for a tensor. On the other hand, bind_arg has the same use, but for non-tensor values. Note that there are no actual tensor values being inputted here, just placeholders.


Example of task invocation method:  
 ```cpp
OpTaskInvocation forward(MultiHeadAttentionAttrs const &attrs) {

OpTaskBinding b; //creating the binding


b.bind(QUERY, input_tensor(0)); //tells us the QUERY attribute will be a input_tensor type

b.bind(KEY, input_tensor(1));

b.bind(VALUE, input_tensor(2));

b.bind(WEIGHTS, weight_tensor(0));

b.bind(OUTPUT, output_tensor(0));

  

b.bind_arg(PROFILING, profiling_settings());

b.bind_arg(PER_DEVICE_STATE, per_device_op_state<MHAPerDeviceState>());

  
return {ATTENTION_FWD_TASK_ID, b};

}
```
  
  

**Init:**
In this method, the INIT_TASK_ID needs to be associated with its binding and returned. The init task binding needs to have the below:

-Handle & Attributes
-Query, Key, Value tensor shapes
-Query, Key, Value tensor sizes


**Forward:**
In this method, the FWD_TASK_ID needs to be associated with its binding and returned. The forward task binding needs to have the below, which are required for forward propagation:

  
-Profiling and Per Device State attributes
-Query, Key, Value input tensors
-Weights tensor
-Output tensor

  

**Backward:**
In this method, the BWD_TASK_ID needs to be associated with its binding and returned. The forward task binding needs to have the below, which are required for backward propagation:

  

Backward propagation bindings can be inferred from the forward propagation bindings, meaning that it is understood that the same attributes - inputs, tensor sizes, etc - will be used for back propagation.

  
```cpp
OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);
```
-----------------------------------------------------------------------------------------------------
 

## Implement the task impl (implementation) functions for the init, forward, and backward task and their corresponding wrapper functions (return a + b)

**Task Implementations: launches the kernel**

The implementation function is the actual body of the function (ie. What it does with the parameters inputted) and can be comparable to the return a + b part of our demo function. Inside the implementation function (impl), all the parameters are extracted from the accessor, and the operator kernel is launched, along with all of its required tensors.

---------------------------------------------------------------------------------------------------------------------  
Examples of impl function header statements - the only parameter is the accessor which contains all of our task attributes:
```cpp

static DeviceSpecific<MHAPerDeviceState> init_task_impl(TaskArgumentAccessor const &acc)

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc)

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc)

```
  Setting up the accessor and launching the kernel:
```cpp
# using accessor to extract the parameters needed for the kernel launch

auto query = acc.get_tensor<Permissions::RO>(QUERY);

auto key = acc.get_tensor<Permissions::RO>(KEY);

auto value = acc.get_tensor<Permissions::RO>(VALUE);

auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);

auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

.............. 

# Launching forward kernel

return profile(forward_kernel,

	profiling,

	per_device_state,

	query.get_float_ptr(),

	key.get_float_ptr(),

	value.get_float_ptr(),

	weight.get_float_ptr(),

	output.get_float_ptr());

}
```
---------------------------------------------------------------------------------------------------------------------

**Wrapper Function: understandable by Legion**
INIT_TASK/FORWARD_TASK/BWD_TASK is a wrapper over the function body (the task_impl) which makes the body readable by the backing legion (ie. Puts the data in a format which is readable by legion, by using regions, context, and runtime).

---------------------------------------------------------------------------------------------------------------------

  
```cpp
init_task(Task const *task,

std::vector<PhysicalRegion> const &regions,

Context ctx,

Runtime *runtime) {

#creates acc in a format that’s viable with legion & calls the function body (task_impl)

TaskArgumentAccessor acc(task, regions, ctx, runtime);

return init_task_impl(acc);

}
```
  

Register  each of the tasks using their corresponding task id, and provide task signature functions (defining the function)

  

**SIGNATURE:**
Essentially, the function signature represents the f(a, b) in our example function. Just like a normal function signature, this method (fwd_signature, bwd_signature, init_signature) is signifying exactly what parameters are needed for this function. The signature adds these attributes / values / arguments / types of arguments that this task expects, and then returns this task signature instance.

  
```cpp
template <>

OpTaskSignature fwd_signature<ATTENTION_FWD_TASK_ID>() {

	OpTaskSignature fwd(OpTaskType::FWD);

# This function tells us that the parameters required for fwd_task are

# QUERY, KEY, VALUE, WEIGHTS, OUTPUTS, PROFILING and PER_DEVICE_STATE

	fwd.add_input_slot(QUERY);

	fwd.add_input_slot(KEY);

	fwd.add_input_slot(VALUE);

	fwd.add_weight_slot(WEIGHTS);

	fwd.add_output_slot(OUTPUT);

  

	fwd.add_arg_slot<ProfilingSettings>(PROFILING);
	fwd.add_unchecked_arg_slot<MHAPerDeviceState>(PER_DEVICE_STATE);

  return fwd;
}
```  

**REGISTER_TASK:**

We can think of the REGISTER_TASK methods as def in the line def f(a, b). Essentially, we are alerting the backing that this task now exists and needs to be supplied inputs in accordance with its task signature.

  

For the backing to use the operator, the operator needs to register its tasks with the backing. Generally, three tasks need to be registered, using register_task. The three tasks are as follows:

  

Example function definition in attention.cc:

```cpp
void register_task<ATTENTION_INIT_TASK_ID>()

void register_task<ATTENTION_FWD_TASK_ID>()

void register_task<ATTENTION_BWD_TASK_ID>()

  

Each of these register task methods registers the corresponding task function, along with its ID, name, and signature.

  

Ex:

register_task(ATTENTION_FWD_TASK_ID,

"Attention Fwd",

fwd_signature<ATTENTION_FWD_TASK_ID>(),

forward_task);

}
```

  

---------------------------------------------------------------------------------------------------------------------

  

## Finally, provide a profiling calculations method

  

**OPERATOR_COST:**

Every operator has a cost function, called measure_operator_cost. This method must invoke forward and backward implementation functions and collect time and memory profile data, using which metrics calculations are done and returned.

  

float forward_time = forward_task_impl(fwd_accessor).value();

float backward_time = backward_task_impl(bwd_accessor).value();

  

float sync_time = default_estimate_sync_time(env);

return make_metrics(forward_time, backward_time, sync_time, env);

---------------------------------------------------------------------------------------------------------------------

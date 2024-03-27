# OpTaskSpec

The `task_spec` interface is primarily utilized by FlexFlow operators via `OpTaskSignature` and `OpTaskInvocation`. Here, we will describe this usage.

Each operator can execute the following tasks: `forward`, `backward`, and an optional `init` task, each of which invoke different kernel functions. Since `OpTaskSpec` handles the Legion interface, the operator is only responsible for specifying each task's signature and invocation.

## `OpTaskSignature`

An operator must specify
1. An enum of `Slots` that encompass all required tensors and arguments for all of its tasks. 
2. A `fwd_signature<>`, `bwd_signature<>`, and optionally an `init_signature<>` where the template argument is the relevant `task_id_t` (e.g., `ATTENTION_FWD_TASK_ID`).
3. Relevant tensor and argument slots within the signature definition. `OpTaskSignature` provides differentiation between different kinds of tensor and argument slots (e.g., `add_input_slot()` vs `add_weight_slot()`).

## `OpTaskInvocation`

For each `OpTaskSignature` that an operator defines, there should be at least one corresponding `OpTaskInvocation`.

An `OpTaskInvocation` is a wrapper around `{task_id_t, OpTaskBinding}`. A binding (for lack of a better word) *binds* an operator slot to a slot argument type (e.g., `OpTensorSpec`, `ConcreteArgSpec`, etc.). This will eventually generate a `TaskArgumentAccessor` that is passed to the actual `forward`, `backward`, and `init` functions that call the kernel. 

From the `TaskArgumentAccessor`, we can call `get_tensor<>(slot_id)` or `get_argument<>(slot_id)` where the template type is the correct permissions or the correct argument type and pass in the slot ID that we want to access. This will return a `GenericTensorAccessor` that can be passed to the kernel function.

Finally, an operator must `register_task<task_id_t>` with the correct signature and the corresponding function call. 

# Questions:

> For each `OpTaskSignature` that an operator defines, there should be at least one corresponding `OpTaskInvocation`.

Is this always true? 

> This will eventually generate a `TaskArgumentAccessor` that is passed to the actual `forward`, `backward`, and `init` functions that call the kernel. 

In my head the workflow is something like this:
- `TaskArgumentAccessor` is initialized with the relevant legion regions and the legion task information. 
- We should call the appropriate `OpTaskInvocation` in this initializer. Then, `TaskArgumentAccessor` can maintain a `TaskArgumentFormat` which maps the slot ids in our abstraction to specific legion regions? ***I think I'm confused about this a little bit.***
- Then, in the operator file, we can just `get_tensor<>()` and `TaskArgumentAccessor` will return a `GenericTensorAccessor` to pass to the kernel.

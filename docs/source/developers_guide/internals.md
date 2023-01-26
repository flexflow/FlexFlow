# FlexFlow Internals

## The Parallel Computation Graph (PCG)

FlexFlow uses a _Parallel Computation Graph (PCG)_ to simultaneously represent tensor operations, as well as parallelism choices and data movement across nodes. 

### Tensor representations

There are two types of tensor representations in FlexFlow: a [Tensor](./cuda_api/de/da9/structFlexFlow_1_1TensorBase.html) and a [ParallelTensor](./cuda_api/d3/dfc/structFlexFlow_1_1ParallelTensorBase.html). The first variant is used when writing a FlexFlow DNN program, whereas the second is used by the runtime to run all the computations in a distributed fashion. `Tensor` and `ParallelTensor` are implemented as typedef-ed pointers to, respectively, the `TensorBase` (defined in `include/flexflow/tensor.h`) and `ParallelTensorBase` (defined in `include/flexflow/parallel_tensor.h`) structs. 

The `ParallelTensor` struct contains all the information that a `Tensor` also stores, but in addition, it also codifies how the tensor should be parallelized. For instance, a ParallelTensor records how each dimension is *partitioned*, how many *replicas* of the tensors have been created, and the *mapping* between the partitions of the tensors and the physical machines that will store them. 

## Transformation generation

## Joint optimization

#include "task_argument_accessor.h"

using namespace FlexFlow {

  template <typename T>
  T const &LocalTaskArgumentAccessor::get_argument(slot_id slot) const {
    if (slot == PROFILING) {
      return get<ProfilingSettings>(this->arg_bindings.at(slot));
    }
    elif (slot == ATTRS) {
      return get<AggregateAttrs>(this->arg_bindings.at(slot));
    }
    else {
      throw std::runtime_error(
          "Unknown Slot ID in LocalTaskArgumentAccessor::get_argument");
    }
  }

  void *LocalTaskArgumentAccessor::allocate(size_t size) {
    void *ptr =
        local_allocator.allocate(size); // Note: how(when) to free this memory?
    void *cpu_ptr = malloc(size);
    memory_usage += size; // update the usage of memory
    memset(cpu_ptr, 0, size);
    checkCUDA(
        cudaMemcpy(ptr, cpu_ptr, size, cudaMemcpyHostToDevice)); // fill ptr
    free(cpu_ptr);
    return ptr;
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> LocalTaskArgumentAccessor::get_tensor(
      slot_id slot) const {
    SimTensorSpec const &spec = this->tensor_shape_bindings.at(slot);
    if (slot == GATE_PREDS || slot == GATE_ASSIGN) {
      InputParallelTensorDesc gate_preds = get<InputParallelTensorDesc>(
          this->sim_task_binding->tensor_shape_bindings.at(slot));
      DataType data_type = gate_preds.shape.data_type;
      ArrayShape array_shape = {
          gate_preds.shape.dims.get_dims()}; // gate_preds.shape.dims.get_dims()
                                             // return std::vector<size_t>
      size_t shape_size = gate_preds.shape.dims.get_volume() * size_of(shape);
      void *ptr = allocate(shape_size);
      return gate_preds_accessor{data_type, array_shape, ptr};
    } else if (slot == OUTPUT) {
      ParallelTensorShape output_shape = get<ParallelTensorShape>(
          this->sim_task_binding->tensor_shape_bindings.at(slot));
      Datatype data_type = output_shape.data_type;
      ArrayShape array_shape = {
          output_shape.dims.get_dims()}; // output_shape.dims.get_dims() return
                                         // std::vector<size_t>
      size_t shape_size = output_shape.dims.get_volume() * size_of(data_type);
      void *ptr = allocate(shape_size);
      return {data_type, array_shape, ptr};
    } else {
      throw mk_runtime_error(
          "Unknown Slot ID in LocalTaskArgumentAccessor::get_tensor");
    }
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>> get_variadic_tensor(
      slot_id slot) const override {
    std::vector<privilege_mode_to_accessor<PRIV>> result;
    InputVariadicParallelTensorDesc const &spec =
        get<InputVariadicParallelTensorDesc>(
            this->sim_task_binding->tensor_shape_bindings.at(slot));
    for (auto const &shape : spec.shapes) {
      ArrayShape array_shape = {
          shape.dims
              .get_dims()}; // shape.dims.get_dims() return std::vector<size_t>
      size_t shape_size = shape.dims.get_volume() * size_of(shape.data_type);
      void *ptr = allocate(shape_size);
      DataType data_type = shape.data_type;
      result.push_back({data_type, array_shape, ptr});
    }
    return result;
  }

} // namespace FlexFlow
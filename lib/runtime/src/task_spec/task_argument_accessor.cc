#include "task_argument_accessor.h"

using namespace FlexFlow {

template <typename T> T const & LocalTaskArgumentAccessor::get_argument(slot_id slot)  const{
    if(slot == PROFILING) {
        return get<ProfilingSettings>(this->arg_bindings.at(slot));
    } elif (slot == ATTRS) {
        return get<AggregateAttrs>(this->arg_bindings.at(slot));
    } else {
        throw std::runtime_error("Unknown Slot ID in LocalTaskArgumentAccessor::get_argument");
    }
}

template <Permissions PRIV> privilege_mode_to_accessor<PRIV> LocalTaskArgumentAccessor::get_tensor(slot_id slot) const {
    SimTensorSpec const & spec = this->tensor_shape_bindings.at(slot);
    if(slot == GATE_PREDS) {
        InputParallelTensorDesc  gate_preds = get<InputParallelTensorDesc>(this->sim_task_binding->tensor_shape_bindings.at(slot));
        DataType shape = gate_preds.shape;
        ArrayShape = {gate_preds.shape.dims.get_dims()};//gate_preds.shape.dims.get_dims() return std::vector<size_t>
        //TODO: 1)allocate memory for ptr 2)fill ptr 
        //question: 1) how much memory should I allocate? 2) how to fill ptr?
        
        //use gate_preds.shape to get the ArrayShape
        //TODO, I should convert gate_pred to privilege_mode_to_accessor<PRIV>
        NOT_IMPLEMENTED();
    } else if(slot == GATE_ASSIGN) {
        InputVariadicParallelTensorDesc  gate_assign = get<InputVariadicParallelTensorDesc>(this->sim_task_binding->tensor_shape_bindings.at(slot));
        //TODO, I should convert gate_assign to privilege_mode_to_accessor<PRIV>
        NOT_IMPLEMENTED();
    } else if(slot == EXP_PREDS) {
        InputVariadicParallelTensorDesc exp_preds = get<InputVariadicParallelTensorDesc>(this->sim_task_binding->tensor_shape_bindings.at(slot));
        //TODO, I should convert exp_preds to privilege_mode_to_accessor<PRIV>
        NOT_IMPLEMENTED();
    } else if(slot == OUTPUT) {
        ParallelTensorShape  output_shape = get<ParallelTensorShape>(this->sim_task_binding->tensor_shape_bindings.at(slot));
        //TODO, I should convert output_shape to privilege_mode_to_accessor<PRIV>
        NOT_IMPLEMENTED();
    } else {
        throw std::runtime_error("Unknown Slot ID in LocalTaskArgumentAccessor::get_tensor");
    }
//     NOT_IMPLEMENTED();//TODO, I should convert spec to privilege_mode_to_accessor<PRIV>
//lib/runtime/src/accessor.h
//     template <>
// struct privilege_mode_to_accessor_t<Permissions::RW> {
//   using type = GenericTensorAccessorW;
// };

// template <>
// struct privilege_mode_to_accessor_t<Permissions::RO> {
//   using type = GenericTensorAccessorR;
// };

// template <>
// struct privilege_mode_to_accessor_t<Permissions::WO> {
//   using type = GenericTensorAccessorW;
// };

// template <Permissions PRIV>
// using privilege_mode_to_accessor =
//     typename privilege_mode_to_accessor_t<PRIV>::type;
}

}//namespace FlexFlow
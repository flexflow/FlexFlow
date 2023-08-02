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
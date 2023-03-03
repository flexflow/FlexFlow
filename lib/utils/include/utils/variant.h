#ifndef _FLEXFLOW_UTILS_VARIANT_H
#define _FLEXFLOW_UTILS_VARIANT_H 

#include "mpark/variant.hpp"

namespace FlexFlow {

template <class ...Args>
struct variant_join_helper;

using mpark::variant;
using mpark::get;
using mpark::holds_alternative;
using mpark::visit;

template <class ...Args1, class ...Args2>
struct variant_join_helper<variant<Args1...>, variant<Args2...>> {
    using type = variant<Args1..., Args2...>;
};

template <class Variant1, class Variant2>
using variant_join = typename variant_join_helper<Variant1, Variant2>::type;

}

#endif 

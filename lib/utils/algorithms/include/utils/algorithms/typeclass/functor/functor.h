#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_FUNCTOR_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_FUNCTOR_H

#include <functional>
#include <type_traits>
#include "utils/backports/type_identity.h" 

namespace FlexFlow {

/* struct opaque_input_type_t { }; */
/* struct opaque_output_type_t { }; */



/* template < */
/*   typename Instance, */ 
/*   typename B = opaque_output_type_t, */
/*   typename F_B = typename Instance::template F<B>, */ 
/*   typename A = typename Instance::A, */
/*   typename F_A = typename Instance::template F<A>> */
/* struct is_valid_functor_instance */
/*   : std::is_invocable< */
/*       decltype(Instance::template fmap<std::function<B(A)>>), */
/*       F_A, */ 
/*       std::function<B(A)> */
/*       > { }; */

/* template <typename Instance> */
/* inline constexpr bool is_valid_functor_instance_v = is_valid_functor_instance<Instance>::value; */

/* /1* template <typename T, typename Instance, typename InputType = other_functor_opaque_t, typename OutputType = functor_opaque_t> *1/ */
/* /1* struct is_valid_functor_instance2 *1/ */
/* /1*   : std::is_same< *1/ */
/* /1*       std::invoke_result_t<decltype(Instance::template fmap<InputType>), T<InputType> const &, std::function<OutputType(InputType const &)>>, *1/ */ 
/* /1*       T<OutputType> *1/ */
/* /1*     > { }; *1/ */

/* // template <template <typename...> typename T> */
/* // struct default_functor_t { }; */

/* template <typename T, typename Enable = void> */
/* struct default_functor { }; */

/* template <typename T> */
/* using default_functor_t = typename default_functor<T>::type; */

/* template <typename T, typename Functor = default_functor_t<T>> */
/* struct element_type : type_identity<typename Functor::A> { }; */

/* template <typename T, typename Functor = default_functor_t<T>> */
/* using element_type_t = typename element_type<T, Functor>::type; */

/* template <typename T, */ 
/*           typename F, */ 
/*           typename Instance = default_functor_t<T>, */ 
/*           typename = std::enable_if_t<is_valid_functor_instance_v<Instance>> > */
/* auto fmap(T const &t, F const &f) */ 
/* { */
/*   return Instance::template fmap<F>(t, f); */
/* } */

} // namespace FlexFlow

#endif

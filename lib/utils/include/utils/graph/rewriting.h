#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_REWRITING_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_REWRITING_H

#include "labelled_graphs.h"

namespace FlexFlow {

template <typename I,
          typename F,
          typename O = decltype(std::declval<F>()(std::declval<I>()))>
NodeLabelledMultiDiGraph<O> rewrite(NodeLabelledMultiDiGraph<I> const &,
                                    F const &f);

template <typename IN,
          typename IE,
          typename F,
          typename ON = decltype(std::declval<F>()(std::declval<IN>())),
          typename OE = decltype(std::declval<F>()(std::declval<IE>()))>
LabelledMultiDiGraph<ON, OE> rewrite(LabelledMultiDiGraph<IN, IE> const &,
                                     F const &f);

template <typename IN,
          typename IE,
          typename F,
          typename ON = decltype(std::declval<F>()(std::declval<Node const &>(),
                                                   std::declval<IN>())),
          typename OE = decltype(std::declval<F>()(
              std::declval<MultiDiEdge const &>(), std::declval<IE>()))>
OutputLabelledMultiDiGraph<ON, OE>
    rewrite(OutputLabelledMultiDiGraph<IN, IE> const &, F const &f);

} // namespace FlexFlow

#endif

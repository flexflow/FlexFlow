#include "compiler/machine_mapping/split_sp_decomposition.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/transform.h"
#include "utils/variant.h"

namespace FlexFlow {

std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    split_sp_decomposition(SerialSplit const &serial) {
  if (serial.children.size() == 2) {
    return {widen<SerialParallelDecomposition>(serial.children[0]),
            widen<SerialParallelDecomposition>(serial.children[1])};
  }
  SerialSplit decompn1 = serial;
  decompn1.children.pop_back();
  return {SerialParallelDecomposition(decompn1),
          widen<SerialParallelDecomposition>(serial.children.back())};
}

std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    split_sp_decomposition(ParallelSplit const &parallel) {
  if (parallel.children.size() == 2) {
    std::vector<SerialParallelDecomposition> children =
        transform(as_vector(parallel.children), [&](auto const &child) {
          return widen<SerialParallelDecomposition>(child);
        });
    return {children[0], children[1]};
  }
  ParallelSplit decompn1 = parallel;
  std::variant<SerialSplit, Node> child = *parallel.children.begin();
  decompn1.children.erase(child);
  return {SerialParallelDecomposition(decompn1),
          widen<SerialParallelDecomposition>(child)};
}

} // namespace FlexFlow

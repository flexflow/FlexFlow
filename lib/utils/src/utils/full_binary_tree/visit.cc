#include "utils/full_binary_tree/visit.h"

namespace FlexFlow {

template 
  int visit(std::string const &, 
            FullBinaryTreeImplementation<std::string, int, int> const &,
            FullBinaryTreeVisitor<int, std::string, int, int> const &);

} // namespace FlexFlow

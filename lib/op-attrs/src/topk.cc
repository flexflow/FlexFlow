#include "op-attrs/ops/topk.h"

namespace FlexFlow {

TopKAttrs::TopKAttrs(int _k, bool _sorted) : k(_k), sorted(_sorted) {}

} // namespace FlexFlow

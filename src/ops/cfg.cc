#include "flexflow/ops/cfg.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/layer.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::Future;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

Tensor FFModel::cfg(const Tensor input, float guidance_scale) {
    //TODO, do one forward computation
    //First, we get the logits, next_token_logits = input.logits[:, -1, :]
    //Then, we use the input and next_token_logits to call logits_processor
    

    // next_token_logits = outputs.logits[:, -1, :]

    // # pre-process distribution
    // next_tokens_scores = logits_processor(input_ids, next_token_logits) #调用logits_processor，对input_ids和next_token_logits进行处理
    NOT_IMPLEMENTED();
}

} // namespace FlexFlow
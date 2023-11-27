#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_INITIALIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_INITIALIZER_H

#include "op-attrs/datatype.h"
#include "utils/required.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct GlorotUniform {
  req<int> seed;
  /* float scale; */
  /* DataType data_type; */
};
FF_VISITABLE_STRUCT(GlorotUniform, seed);
FF_VISIT_FMTABLE(GlorotUniform);
CHECK_FMTABLE(GlorotUniform);

struct ZeroInitializer {
  ZeroInitializer() = default;
};
FF_VISITABLE_STRUCT(ZeroInitializer);
FF_VISIT_FMTABLE(ZeroInitializer);
CHECK_FMTABLE(ZeroInitializer);

struct UniformInitializer {
  req<int> seed;
  req<float> min_val;
  req<float> max_val;
};
FF_VISITABLE_STRUCT(UniformInitializer, seed, min_val, max_val);
FF_VISIT_FMTABLE(UniformInitializer);
CHECK_FMTABLE(UniformInitializer);

struct NormInitializer {
  req<int> seed;
  req<float> mean;
  req<float> stddev;
};
FF_VISITABLE_STRUCT(NormInitializer, seed, mean, stddev);
FF_VISIT_FMTABLE(NormInitializer);
CHECK_FMTABLE(NormInitializer);

struct ConstantInitializer {
  req<DataTypeValue> value;
};
FF_VISITABLE_STRUCT(ConstantInitializer, value);
FF_VISIT_FMTABLE(ConstantInitializer);
CHECK_FMTABLE(ConstantInitializer);

using Initializer = variant<GlorotUniform,
                            ZeroInitializer,
                            UniformInitializer,
                            NormInitializer,
                            ConstantInitializer>;
CHECK_WELL_BEHAVED_VALUE_TYPE(Initializer);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::Initializer> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::Initializer initializer, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view s = "unknown";
    if (auto const *g = get_if<GlorotUniform>(&initializer)) {
      s = fmt::to_string(*g);
    } else if (auto const *z = get_if<ZeroInitializer>(&initializer)) {
      s = fmt::to_string(*z);
    } else if (auto const *u = get_if<UniformInitializer>(&initializer)) {
      s = fmt::to_string(*u);
    } else if (auto const *n = get_if<NormInitializer>(&initializer)) {
      s = fmt::to_string(*n);
    } else if (auto const *c = get_if<ConstantInitializer>(&initializer)) {
      s = fmt::to_string(*c);
    }
    return formatter<string_view>::format(s, ctx);
  }
};

} // namespace fmt

#endif

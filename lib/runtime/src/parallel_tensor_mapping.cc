#include "parallel_tensor_mapping.h"

using namespace Legion;

namespace FlexFlow {

template <int NDIM, int TDIM>
void map_linear_weight(ParallelTensor weight,
                       Op const *op,
                       LegionConfig const &config,
                       CompMode computationMode) {
  using namespace Legion;

  assert(op->op_type == OP_LINEAR);
  std::string pcname = op->name;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, op->parallel_is);
  int num_parts[TDIM];
  for (int i = 0; i < TDIM; i++) {
    num_parts[i] = part_rect.hi[i] - part_rect.lo[i] + 1;
  }
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (weight->data_type) {
    case DT_FLOAT:
      allocator.allocate_field(sizeof(float), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(double), FID_DATA);
      break;
    case DT_INT32:
      allocator.allocate_field(sizeof(int), FID_DATA);
      break;
    default:
      assert(false);
  }
  int out_channels = weight->dims[weight->num_dims - 1].size;
  // Step 1: forward region and partition
  if (weight->sync_type == ParameterSyncType::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    assert(out_channels % num_parts[0] == 0);
    hi[NDIM - 1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels / num_parts[0];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else if (weight->sync_type == ParameterSyncType::NCCL) {
    // FIXME: Currently only support the sample dimension for operators with
    // NCCL
    // for (int i = 0; i < TDIM-1; i++)
    //  assert(num_parts[i] == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++) {
      num_batches *= num_parts[i];
    }
    hi[NDIM - 1] = num_batches * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels / num_parts[0];
    for (int i = 1; i < TDIM; i++) {
      transform[NDIM - 1][i] = transform[NDIM - 1][i - 1] * num_parts[i - 1];
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else {
    assert(false);
  }
  // Step 2: initialize region
  if (weight->initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    weight->initializer->init(config, weight);
  }
  // Step 3: backward region
  if (weight->create_gradients && computationMode == COMP_MODE_TRAINING) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++) {
      num_batches *= num_parts[i];
    }
    hi[NDIM - 1] = num_batches * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels / num_parts[0];
    for (int i = 1; i < TDIM; i++) {
      transform[NDIM - 1][i] = transform[NDIM - 1][i - 1] * num_parts[i - 1];
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part_grad =
        runtime->get_logical_partition(ctx, weight->region_grad, ip);
  }
}

template <int NDIM>
void map_conv_weight(ParallelTensor weight,
                     Op const *op,
                     LegionConfig const &config,
                     CompMode computationMode) {
  using namespace Legion;

  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, op->parallel_is);
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  // Currently assume we do not split over the channel dimension
  assert(num_par_c == 1);
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (weight->data_type) {
    case DT_FLOAT:
      allocator.allocate_field(sizeof(float), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(double), FID_DATA);
      break;
    case DT_INT32:
      allocator.allocate_field(sizeof(int), FID_DATA);
      break;
    default:
      assert(false);
  }
  // Step 1: forward region and partition
  int out_channels = weight->dims[weight->num_dims - 1].size;
  if (weight->sync_type == ParameterSyncType::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < 4; j++) {
        transform[i][j] = 0;
      }
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, rect);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else if (weight->sync_type == ParameterSyncType::NCCL) {
    // Currently only support sample and attribute parallelism for NCCL
    // communication
    assert(num_par_c == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    hi[NDIM - 1] = num_par_n * num_par_h * num_par_w * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < 4; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels;
    transform[NDIM - 1][1] = out_channels * num_par_w;
    transform[NDIM - 1][2] = out_channels * num_par_w * num_par_h;
    transform[NDIM - 1][3] = out_channels * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else {
    // Unsupported Parameter type
    assert(false);
  }
  // Step 2: initialize region
  if (weight->initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    weight->initializer->init(config, weight);
  }
  // Step 3: backward regin and partition
  if (weight->create_gradients && computationMode == COMP_MODE_TRAINING) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    hi[NDIM - 1] = num_par_n * num_par_h * num_par_w * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < 4; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels;
    transform[NDIM - 1][1] = out_channels * num_par_w;
    transform[NDIM - 1][2] = out_channels * num_par_w * num_par_h;
    transform[NDIM - 1][3] = out_channels * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part_grad =
        runtime->get_logical_partition(ctx, weight->region_grad, ip);
  }
}

template <int NDIM>
void map_weight_with_dim(ParallelTensor &weight,
                         Op const *parallel_op,
                         LegionConfig const &config,
                         CompMode computationMode) {
  // Step 0: check we are the owner or the owner is NULL
  // in which case set the owner to us
  if (weight->owner_op == NULL) {
    weight->owner_op = parallel_op;
    weight->owner_idx = -1; // meaning tensor is not an output of op
  } else {
    assert(weight->owner_op == parallel_op);
  }
  assert(parallel_op != NULL);
  int tdim = parallel_op->outputs[0]->num_dims;
  switch (parallel_op->op_type) {
    case OP_LINEAR:
    case OP_EMBEDDING:
    case OP_MULTIHEAD_ATTENTION: {
      switch (tdim) {
#define DIMFUNC(TDIM)                                                          \
  case TDIM: {                                                                 \
    map_linear_weight<NDIM, TDIM>(                                             \
        weight, parallel_op, config, computationMode);                         \
    break;                                                                     \
  }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default: {
          assert(false);
        }
      }
      break;
    }
    case OP_CONV2D:
    case OP_BATCHNORM: {
      map_conv_weight<NDIM>(weight, parallel_op, config, computationMode);
      break;
    }
    default: {
      fprintf(stderr,
              "FlexFlow currently does not support this weight"
              "type (%d). Report the error to the FlexFlow team.\n",
              parallel_op->op_type);
      assert(false && "Unsupported type for mapping weight");
    }
  }
}

void map_weight(ParallelTensor &weight,
                Op const *op,
                LegionConfig const &config,
                CompMode computationMode) {
  switch (weight->num_dims) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    map_weight_with_dim<DIM>(weight, op, config, computationMode);             \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dim
      assert(false);
    }
  }
}

template <int NDIM, int TDIM>
void create_data_parallel_partition_with_diff_dims(
    ParallelTensor const &tensor,
    Legion::IndexSpaceT<TDIM> const &part_is,
    Legion::LogicalPartition &part_fwd,
    Legion::LogicalPartition &part_bwd,
    LegionConfig const &config,
    CompMode computationMode) {
  using namespace Legion;

  assert(tensor->num_dims == NDIM);
  if (computationMode == COMP_MODE_TRAINING) {
    // Current assume forward and grad share the same index space
    if (tensor->region_grad != LogicalRegion::NO_REGION) {
      assert(tensor->region.get_index_space() ==
             tensor->region_grad.get_index_space());
    }
  }
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, tensor->region.get_index_space());
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  // Assume it is data parallel
  for (int i = 0; i < TDIM - 1; i++) {
    assert(part_rect.lo[i] == part_rect.hi[i]);
  }
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  for (int i = 0; i < NDIM; i++) {
    int nparts = 1;
    if (i == NDIM - 1) {
      nparts = part_rect.hi[TDIM - 1] - part_rect.lo[TDIM - 1] + 1;
    }
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < TDIM; j++) {
      transform[i][j] = 0;
    }
  }
  transform[NDIM - 1][TDIM - 1] = extent.hi[NDIM - 1] - extent.lo[NDIM - 1] + 1;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, tensor->region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor->region, ip);
  if (computationMode == COMP_MODE_TRAINING) {
    if (tensor->region_grad != LogicalRegion::NO_REGION) {
      part_bwd = runtime->get_logical_partition(ctx, tensor->region_grad, ip);
    }
  } else {
    part_bwd = LogicalPartition::NO_PART;
  }
}

template <int NDIM>
void create_disjoint_partition(ParallelTensor const &tensor,
                               Legion::IndexSpaceT<NDIM> const &part_is,
                               Legion::LogicalPartition &part_fwd,
                               Legion::LogicalPartition &part_bwd,
                               LegionConfig const &config) {
  using namespace Legion;

  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Check that dimension sizes match
  {
    assert(tensor->num_dims == NDIM);
    Domain domain = runtime->get_index_space_domain(ctx, part_is);
    assert(domain.get_dim() == NDIM);
  }
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, tensor->region.get_index_space());
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, NDIM> transform;
  Point<NDIM> ext_hi;
  for (int i = 0; i < NDIM; i++) {
    int nparts = part_rect.hi[i] - part_rect.lo[i] + 1;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < NDIM; j++) {
      if (i == j) {
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      } else {
        transform[i][j] = 0;
      }
    }
  }
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, tensor->region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor->region, ip);
  if (tensor->region_grad != LogicalRegion::NO_REGION) {
    // Current assume forward and grad share the same index space
    assert(tensor->region.get_index_space() ==
           tensor->region_grad.get_index_space());
    part_bwd = runtime->get_logical_partition(ctx, tensor->region_grad, ip);
  } else {
    part_bwd = LogicalPartition::NO_PART;
  }
}

template <int NDIM, int TDIM>
void create_aliased_partition_with_dim2(
    const ParallelDim dims[],
    int aliased_dim,
    Legion::IndexSpaceT<TDIM> const &part_is,
    Legion::LogicalRegion const &region,
    Legion::LogicalPartition &part,
    LegionConfig const &config) {
  using namespace Legion;

  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, region.get_index_space());
  for (int i = 0; i < NDIM; i++) {
    int nparts = dims[i].degree;
    if (aliased_dim == i) {
      nparts = 1;
    }
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < TDIM; j++) {
      if (dims[i].parallel_idx == j && i != aliased_dim) {
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      } else {
        transform[i][j] = 0;
      }
    }
  }
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, region.get_index_space(), part_is, transform, extent);
  // assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part = runtime->get_logical_partition(ctx, region, ip);
}

void create_aliased_partition(int num_dims,
                              const ParallelDim dims[],
                              int aliased_dim,
                              Legion::IndexSpace const &part_is,
                              Legion::LogicalRegion const &region,
                              Legion::LogicalPartition &part,
                              LegionConfig const &config) {
  using namespace Legion;

  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Domain task_domain = runtime->get_index_space_domain(ctx, part_is);
  switch ((num_dims - 1) * MAX_TENSOR_DIM + task_domain.get_dim() - 1) {
#define DIMFUNC(NDIM, TDIM)                                                    \
  case (NDIM - 1) * MAX_TENSOR_DIM + (TDIM - 1): {                             \
    IndexSpaceT<TDIM> part_is_t(part_is);                                      \
    return create_aliased_partition_with_dim2<NDIM, TDIM>(                     \
        dims, aliased_dim, part_is_t, region, part, config);                   \
  }
    LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported NDIM/TDIM");
  }
}

template <int NDIM, int TDIM>
void create_disjoint_partition_with_dim2(
    const ParallelDim dims[],
    Legion::IndexSpaceT<TDIM> const &part_is,
    Legion::LogicalRegion const &region,
    Legion::LogicalPartition &part,
    LegionConfig const &config) {
  using namespace Legion;

  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, region.get_index_space());
  for (int i = 0; i < NDIM; i++) {
    int nparts = dims[i].degree;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < TDIM; j++) {
      if (dims[i].parallel_idx == j) {
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      } else {
        transform[i][j] = 0;
      }
    }
  }
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part = runtime->get_logical_partition(ctx, region, ip);
}

void create_disjoint_partition(int num_dims,
                               const ParallelDim dims[],
                               Legion::IndexSpace const &part_is,
                               Legion::LogicalRegion const &region,
                               Legion::LogicalPartition &part,
                               LegionConfig const &config) {
  using namespace Legion;

  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Domain task_domain = runtime->get_index_space_domain(ctx, part_is);
  switch ((num_dims - 1) * MAX_TENSOR_DIM + task_domain.get_dim() - 1) {
#define DIMFUNC(NDIM, TDIM)                                                    \
  case (NDIM - 1) * MAX_TENSOR_DIM + (TDIM - 1): {                             \
    IndexSpaceT<TDIM> part_is_t(part_is);                                      \
    return create_disjoint_partition_with_dim2<NDIM, TDIM>(                    \
        dims, part_is_t, region, part, config);                                \
  }
    LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported NDIM/TDIM");
  }
}

template <int NDIM, int TDIM>
void map_tensor_with_dim2(ParallelTensor &tensor,
                          Op const *parallel_op,
                          LegionConfig const &config,
                          IndexSpaceManager &is_mgr,
                          CompMode computationMode) {
  // Step 0: check we are the owner or the owner is NULL
  // in which case set the owner to us
  if (tensor->owner_op == NULL) {
    tensor->owner_op = parallel_op;
    tensor->owner_idx = -1; // meaning tensor is not an output of op
  } else {
    // assert tensor->owner_op == parallel_op or parallel_op == nullptr,
    // which indicates the tensor is not parallelized
    assert(tensor->owner_op == parallel_op || parallel_op == nullptr);
  }
  // Step 1: create regions
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;

  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (tensor->data_type) {
    case DT_HALF:
      allocator.allocate_field(sizeof(half), FID_DATA);
      break;
    case DT_FLOAT:
      allocator.allocate_field(sizeof(float), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(double), FID_DATA);
      break;
    case DT_INT32:
      allocator.allocate_field(sizeof(int32_t), FID_DATA);
      break;
    case DT_INT64:
      allocator.allocate_field(sizeof(int64_t), FID_DATA);
      break;
    default:
      assert(false);
  }

  Point<NDIM> hi;
  for (int i = 0; i < NDIM; i++) {
    hi[i] = tensor->dims[i].size - 1;
  }
  Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  tensor->region = runtime->create_logical_region(ctx, is, fs);
  if (tensor->create_gradients && computationMode == COMP_MODE_TRAINING) {
    tensor->region_grad = runtime->create_logical_region(ctx, is, fs);
  }

  // Step 2: create partitions if parallel_op != NULL
  if (parallel_op != NULL) {
    IndexSpaceT<TDIM> part_is =
        (IndexSpaceT<TDIM>)is_mgr.get_or_create_task_is(tensor->get_shape());
    // Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
    Transform<NDIM, TDIM> transform;
    Point<NDIM> ext_hi;
    for (int i = 0; i < NDIM; i++) {
      int nparts = tensor->dims[i].degree;
      ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
    }
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        if (tensor->dims[i].parallel_idx == j) {
          transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
        } else {
          transform[i][j] = 0;
        }
      }
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
    tensor->part = runtime->get_logical_partition(ctx, tensor->region, ip);
    if (tensor->create_gradients && computationMode == COMP_MODE_TRAINING) {
      tensor->part_grad =
          runtime->get_logical_partition(ctx, tensor->region_grad, ip);
    }
  }
  // Step 3: initialize the tensor
  if (tensor->initializer != NULL) {
    tensor->initializer->init(config, tensor);
  }
}

// Map tensor using parallelization strategies described in parallel_op
template <int NDIM>
void map_tensor_with_dim(ParallelTensor &tensor,
                         Op const *parallel_op,
                         LegionConfig const &config,
                         IndexSpaceManager &is_mgr,
                         CompMode computationMode) {
  tensor->parallel_is = is_mgr.get_or_create_task_is(tensor->get_shape());
  assert(tensor->owner_op != NULL);
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Domain task_domain =
      runtime->get_index_space_domain(ctx, tensor->parallel_is);
  switch (task_domain.get_dim()) {
#define DIMFUNC(TDIM)                                                          \
  case TDIM: {                                                                 \
    map_tensor_with_dim2<NDIM, TDIM>(                                          \
        tensor, parallel_op, config, is_mgr, computationMode);                 \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      assert(false && "Unsupported Task Dim");
    }
  }
}

void map_tensor(ParallelTensor &tensor,
                Op const *op,
                LegionConfig const &config,
                IndexSpaceManager &is_mgr,
                CompMode computationMode) {
  switch (tensor->num_dims) {
#define DIMFUNC(NDIM)                                                          \
  case NDIM: {                                                                 \
    map_tensor_with_dim<NDIM>(tensor, op, config, is_mgr, computationMode);    \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dim
      assert(false);
    }
  }
}

} // namespace FlexFlow

#ifndef _LEGION_CNN_HELPER_H_
#define _LEGION_CNN_HELPER_H_
#include "legion.h"
using namespace Legion;

__global__
void scale_kernel(float* ptr, coord_t size, float a, float b);

#endif

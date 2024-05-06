ff_add_test_executable(
  NAME
    kernel-tests
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE 
    src/
  DEPS
    utils
    compiler
    doctest
    utils-test-common
    kernels
    op-attrs
    pcg
    cuda
    cudnn
    nccl
    cudart
)


# set(project_target kernel-tests)

# project(${project_target} 
# 	LANGUAGES CXX CUDA)

# file(GLOB_RECURSE SRC
#      CONFIGURE_DEPENDS
#      LIST_DIRECTORIES False
#      src/*.cc
#      )

# add_executable(
#   ${project_target}
#   ${SRC}
# )

# target_link_libraries(
#   ${project_target}
#   utils
#   compiler
#   utils-test-common
#   kernels
#   op-attrs
#   pcg
#   cuda
#   cudnn
#   nccl
#   doctest
#   cudart
# )

# target_compile_definitions(${project_target} PRIVATE FF_TEST_SUITE="${project_target}")

# define_ff_vars(${project_target})

# ff_set_cxx_properties(${project_target})

# doctest_discover_tests(${project_target} ADD_LABELS 1)
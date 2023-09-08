ff_add_library(
  NAME
    visitable
  DEPS
    visit_struct
)      

ff_add_intree_test_executable(
  NAME
    visitable-test
  DEPS
    visitable
)

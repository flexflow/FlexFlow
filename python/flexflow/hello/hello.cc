#include "legion.h"

using namespace Legion;

#include "hello.h" 

void launch_hello_world_task()
{
  Runtime *runtime = Runtime::get_runtime();
  Context ctx = Runtime::get_context();
  TaskLauncher launcher(HELLO_WORLD_ID, TaskArgument(NULL, 0));
  runtime->execute_task(ctx, launcher);
}

void hello_world_task(const Task *task, 
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  // A task runs just like a normal C++ function.
  printf("Hello World!\n");
}


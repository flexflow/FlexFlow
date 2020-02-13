enum HelloTaskIDs {
  HELLO_WORLD_ID = 112,
};

void hello_world_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);

#ifdef __cplusplus
extern "C"
{
#endif

void launch_hello_world_task(char *name);

#ifdef __cplusplus
} // extern "C"
#endif

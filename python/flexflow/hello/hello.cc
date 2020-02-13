#include "legion.h"

using namespace Legion;

#include "hello.h" 

#ifndef TYPE_SAFE_LEGATE
#define TYPE_SAFE_LEGATE  false
#endif

class FFSerializer {
public:
  FFSerializer()
    : length(0) { 
    current_ptr = args;
  }
public:
  inline void pack_32bit_int(int32_t val) {
    memcpy((void*)current_ptr, (const void*)&val, sizeof(int32_t));
    length += sizeof(int32_t);
    current_ptr += sizeof(int32_t);
  }
  inline void pack_char(char val) {
    memcpy((void*)current_ptr, (const void*)&val, sizeof(char));
    length += sizeof(char);
    current_ptr += sizeof(char);
  }
  inline void pack_string(std::string val) {
    int32_t len = val.length();
    pack_32bit_int(len);
    const char * c_val = val.c_str();
    for (int i = 0; i < len; i++)
      pack_char(c_val[i]);
  }
public:
  char args[100];
  char* current_ptr;
  size_t length;
};

class FFDeserializer {
public:
  FFDeserializer(const void *a, size_t l)
    : args(static_cast<const char*>(a)), length(l) { }
public:
  inline void check_type(int type_val, size_t type_size) {
    assert(length >= sizeof(int));
    int expected_type = *((const int*)args);
    length -= sizeof(int);
    args += sizeof(int);
    assert(expected_type == type_val);
    assert(length >= type_size);
  }
  inline int32_t unpack_32bit_int(void) {
    if (TYPE_SAFE_LEGATE)
      check_type(2, sizeof(int32_t));
    int32_t result = *((const int32_t*)args);
    length -= sizeof(int32_t);
    args += sizeof(int32_t);
    return result;
  }
  inline char unpack_char(void) {
    if (TYPE_SAFE_LEGATE)
      check_type(11, sizeof(char));
    char result = *((const char*)args);
    length -= sizeof(char);
    args += sizeof(char);
    return result;
  }
  inline std::string unpack_string(void) {
    int size = unpack_32bit_int();
    std::string result;
    for (int i = 0; i < size; i++)
      result.push_back(unpack_char());
    return result;
  }
protected:
  const char *args;
  size_t length;
};

void launch_hello_world_task(char *name)
{
  Runtime *runtime = Runtime::get_runtime();
  Context ctx = Runtime::get_context();
  FFSerializer ser;
  printf("%s\n", name);
  std::string sname = name;
  ser.pack_string(sname);
  TaskLauncher launcher(HELLO_WORLD_ID, TaskArgument(ser.args, ser.length));
  runtime->execute_task(ctx, launcher);
}

void hello_world_task(const Task *task, 
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  std::string name = "default";
  if (task->arglen != 0) {
    //printf("arglen %ld, %s\n", task->arglen, (char*)task->args);
    FFDeserializer derez(task->args, task->arglen);
    name = derez.unpack_string();
  }
  
  printf("Hello World! %s\n", name.c_str());
}


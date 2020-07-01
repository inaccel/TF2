#include "includes.h"

void *alignedMalloc(size_t size) {
  void *result = NULL;
  posix_memalign (&result, 64, size);
  return result;
}

// Sets the current working directory to be the same as the directory
// containing the running executable.
bool setCwdToExeDir() {
  // Get path of executable.
  char path[300];
  ssize_t n = readlink("/proc/self/exe", path, sizeof(path)/sizeof(path[0]) - 1);
  if(n == -1) {
    return false;
  }
  path[n] = 0;

  // Find the last '\' or '/' and terminate the path there; it is now
  // the directory containing the executable.
  size_t i;
  for(i = strlen(path) - 1; i > 0 && path[i] != '/' && path[i] != '\\'; --i);
  path[i] = '\0';

  // Change the current directory.     // Linux
  chdir(path);

  return true;
}

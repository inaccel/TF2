#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include "includes.h"

void *alignedMalloc(size_t size) {
  void *result = NULL;
  if(posix_memalign (&result, 64, size))
    perror("Could not allocate aligned memory.");
  return result;
}

// Function that stores all the filenames found under the specified directroy
// in a std::vector<std::string>.
void read_directory(const std::string& dirName, std::vector<std::string>& v) {
    DIR* dir = opendir(dirName.c_str());
    if (!dir) return;

    struct stat s;
    struct dirent * entry;

    if (lstat(dirName.c_str(), &s) != 0 || !S_ISDIR(s.st_mode)) return;

    while ((entry = readdir(dir)) != NULL) {
        std::string filename = dirName + "/" + entry->d_name;
        if (lstat(filename.c_str(), &s) == 0 && S_ISDIR(s.st_mode)) {
            if ((filename == (dirName + "/.")) || (filename == (dirName + "/..")))
                continue;
            /*if the directory isn't . or ..*/
            else read_directory(filename, v);
        }
        else v.push_back(filename);
    }

    closedir(dir);
}

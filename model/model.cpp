#include "model.h"
#include <iostream>

#include <fcntl.h>     // declares open()
#include <unistd.h>    // declares close()
#include <sys/mman.h>  // declares mmap()
#include <sys/stat.h>


void* Parameters::map_file(const std::string& path){
    // Get file descriptor
    int fd = open(path.c_str(), O_RDONLY);

    if (fd < 0){
        std::cerr << "Model binary open failed" << std::endl;
        std::exit(1);
    }

    // Get the size of the binary file
    struct stat st;
    if (fstat(fd, &st) < 0){
        std::cerr << "Model binary get size failed" << std::endl;
        std::exit(1);
    }

    size_t size = st.st_size;

    // Load the file into virtual memory using mmap
    void* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (p == MAP_FAILED){
        std::cerr << "Model mmap failed" << std::endl;
        std::exit(1);
    }

    close(fd);

    return p;
}



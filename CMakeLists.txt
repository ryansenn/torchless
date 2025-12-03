cmake_minimum_required(VERSION 3.20)
project(torchless CXX)

set(CMAKE_CXX_STANDARD 17)

if(APPLE)
    set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${LIBOMP_ROOT}/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LIBOMP_ROOT}/lib -lomp")
else()
    find_package(OpenMP REQUIRED)
endif()

add_executable(torchless
        main.cpp
        src/backend/cpu/kernels.cpp
        src/common/tensor.cpp
        src/loader/parameters.cpp
        src/model/mistral/modules.cpp
        src/tokenizer/tokenizer.cpp
)

if(NOT APPLE)
    target_link_libraries(torchless PRIVATE OpenMP::OpenMP_CXX)
endif()
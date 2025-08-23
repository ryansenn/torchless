cmake_minimum_required(VERSION 3.16)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file(GLOB MODEL_SRC "model/*.cpp")
file(GLOB TEST_SRC "test/*")

add_executable(main_exec main.cpp math_ops.cpp ${MODEL_SRC}
        math_ops.h)
add_executable(test_exec math_ops.cpp ${MODEL_SRC} ${TEST_SRC}
        math_ops.h)
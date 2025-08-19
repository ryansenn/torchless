cmake_minimum_required(VERSION 3.16)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file(GLOB MODEL_SRC "model/*.cpp")
file(GLOB TEST_SRC "test/*")

add_executable(main_exec main.cpp infer.cpp ${MODEL_SRC})
add_executable(test_exec  infer.cpp ${MODEL_SRC} ${TEST_SRC})
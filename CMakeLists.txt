cmake_minimum_required(VERSION 3.16)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file(GLOB MODEL_SRC "model/*")
file(GLOB INFERENCE_SRC "inference/*")
file(GLOB TEST_SRC "test/*")

add_executable(main_exec main.cpp ${MODEL_SRC} ${INFERENCE_SRC})
add_executable(test_exec ${MODEL_SRC} ${INFERENCE_SRC} ${TEST_SRC})
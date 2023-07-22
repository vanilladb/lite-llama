cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(llama)

find_package(Torch REQUIRED)

add_executable(llama llama.cpp)

target_link_libraries(llama "${TORCH_LIBRARIES}" "-L/usr/local/lib -lsentencepiece -lsentencepiece_train")
set_property(TARGET llama PROPERTY CXX_STANDARD 17)


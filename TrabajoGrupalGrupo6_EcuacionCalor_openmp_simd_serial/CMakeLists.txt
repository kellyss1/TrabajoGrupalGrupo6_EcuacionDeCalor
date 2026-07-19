cmake_minimum_required(VERSION 3.16)

project(EcuacionCalor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -fopenmp" )


find_package(fmt CONFIG REQUIRED)
find_package(SFML COMPONENTS 
                    Network
                    System 
                    Graphics 
                    Window
                    Audio
                    CONFIG REQUIRED)

add_executable(EcuacionCalor 
main.cpp
fractal_calor.cpp
ecuacion_calor_openmp.cpp
fractal_calor_simd.cpp
palette.cpp
)

target_link_libraries(EcuacionCalor PRIVATE 
fmt::fmt 
SFML::System
SFML::Graphics
SFML::Window
SFML::Network
SFML::Audio
)
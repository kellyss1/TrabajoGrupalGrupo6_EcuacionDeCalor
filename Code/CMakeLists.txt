cmake_minimum_required(VERSION 3.16)

project(EcuacionCalor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -fopenmp" )




find_package(fmt CONFIG REQUIRED)
#--config mpi
set(MPI_ROOT "c:/oneAPI/mpi/latest")
set(MPI_INCLUDE "${MPI_ROOT}/include")
set(MPI_LIB "${MPI_ROOT}/lib")
include_directories(   
     ${MPI_INCLUDE}
     )



find_package(SFML COMPONENTS 
                    Network
                    System 
                    Graphics 
                    Window
                    Audio
                    CONFIG REQUIRED)

add_executable(EcuacionCalor 
main.cpp
)

target_link_libraries(EcuacionCalor PRIVATE 
fmt::fmt 
SFML::System
SFML::Graphics
SFML::Window
SFML::Network
SFML::Audio
${MPI_LIB}/impi.lib
)
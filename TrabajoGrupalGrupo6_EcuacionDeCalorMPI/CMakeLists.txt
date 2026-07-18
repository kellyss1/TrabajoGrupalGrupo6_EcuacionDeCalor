cmake_minimum_required(VERSION 3.16)

project(EcuacionCalor)


find_package(fmt CONFIG REQUIRED)
find_package(SFML COMPONENTS 
                    Network
                    System 
                    Graphics 
                    Window
                    Audio
                    CONFIG REQUIRED)



#--config mpi
set(MPI_ROOT "c:/oneAPI/mpi/latest")
set(MPI_INCLUDE "${MPI_ROOT}/include")
set(MPI_LIB "${MPI_ROOT}/lib")
include_directories(   
     ${MPI_INCLUDE}
     )


add_executable(EcuacionCalor 
main.cpp
ecuacion_calor.cpp
palette.cpp
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
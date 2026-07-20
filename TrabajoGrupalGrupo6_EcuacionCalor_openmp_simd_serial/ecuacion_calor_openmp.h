#ifndef ECUACION_CALOR_OPENMP_H
#define ECUACION_CALOR_OPENMP_H

#include <cstdint>

void ecuacion_calor_openmp_regiones(
    uint32_t Nx,               // numero de col
    uint32_t Ny,               // numero de fil
    double Lx,                 // longitud en x
    double Ly,                 // longitud en y
    double alpha,              // difusividad
    double dt,                 // paso de tiempo
    uint32_t max_iter,         // numero maximo de iteraciones
    double tol,                // tolerancia para el criterio de parada
    uint32_t pasos_a_ejecutar, // modo continuo o paso a paso
    uint32_t *pixel_buffer, // color dentro de mi pixel buffer
    uint32_t *iter_out,     // lee la iteracion actual
    double *residuo_out,    // lee el residuo actual
    double *mflops_out,     // lee el rendimiento en MFlops
    uint32_t *threads_out,   // lee el numero de hilos activos
    int num_threads
);

//misma firma pero con AVX2 
void ecuacion_calor_openmp_simd(
    uint32_t Nx,
    uint32_t Ny,
    double Lx,
    double Ly,
    double alpha,
    double dt,
    uint32_t max_iter,
    double tol,
    uint32_t pasos_a_ejecutar,
    uint32_t *pixel_buffer,
    uint32_t *iter_out,
    double *residuo_out,
    double *mflops_out,
    uint32_t *threads_out,
    int num_threads
);



#endif
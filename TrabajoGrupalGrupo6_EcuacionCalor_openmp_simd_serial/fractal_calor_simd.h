#ifndef FRACTAL_CALOR_SIMD_H

#define FRACTAL_CALOR_SIMD_H

#include <cstdint>

void ecuacion_Calor_SIMD(
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
    double *mflops_out      // lee el rendimiento en MFlops
    
);

#endif
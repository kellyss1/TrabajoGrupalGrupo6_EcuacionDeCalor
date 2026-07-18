#ifndef FRACTAL_CALOR_H

#define FRACTAL_CALOR_H

#include <cstdint>

void ecuacionCalorMPI(
    uint32_t Nx,               // numero de col
    uint32_t Ny,               // numero de fil
    double Lx,                 // longitud en x
    double Ly,                 // longitud en y
    double alpha,              // difusividad
    double dt,                 // paso de tiempo
    uint32_t max_iter,         // numero maximo de iteraciones
    double tol,                // tolerancia para el criterio de parada
    uint32_t pasos_a_ejecutar, // modo continuo o paso a paso
    int rank,                  // rank de este proceso
    int nprocs,                // total de procesos
    uint32_t delta,            // filas a trabajar por rank
    uint32_t row_start,        // fila global donde empieza este rank
    uint32_t valid_rows,       // filas reales del rank sin padding
    uint32_t *pixel_buffer,    // color dentro de mi pixel buffer
    uint32_t *iter_out,        // lee la iteracion actual
    double *residuo_out,       // lee el residuo actual
    double *mflops_out,        // validacion en rank 0 de la suma
    double *mflops_total_out

);

#endif
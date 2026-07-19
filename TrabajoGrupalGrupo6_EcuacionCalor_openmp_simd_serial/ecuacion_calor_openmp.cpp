#include "ecuacion_calor_openmp.h"
#include "palette.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

static double stencil(
    const std::vector<double> &u,
    uint32_t i, uint32_t j, uint32_t Nx, double r)
{
    uint32_t centro = j * Nx + i;
    uint32_t derecha = j * Nx + (i + 1);
    uint32_t izquierda = j * Nx + (i - 1);
    uint32_t arriba = (j - 1) * Nx + i;
    uint32_t abajo = (j + 1) * Nx + i;

    return u[centro] +
           r * (u[derecha] + u[izquierda] + u[arriba] + u[abajo] - 4.0 * u[centro]);
}

static void calor_acotado_openmp(
    std::vector<double> &u,
    std::vector<double> &u_new,
    uint32_t Nx, uint32_t Ny,
    double r, double tol,
    uint32_t &iter, double &residuo,
    uint32_t pasos_por_llamada,
    uint32_t max_iter)
{
    uint32_t hechos = 0;
    //variable compartida
    double suma_global = 0.0;


    // region paralela abierta afuera del while para no crear y destruir hilos en cada iteracion
    #pragma omp parallel
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        //reparto manual de las filas internas entre los hilos
        uint32_t filas_internas = Ny - 2;
        uint32_t delta = (filas_internas + thread_count - 1) / thread_count; //techo para evitar casteos
        uint32_t start = 1 + thread_id * delta;
        uint32_t end = start + delta;
        if (end > Ny - 1) end = Ny - 1;
        if (start > Ny - 1) start = Ny - 1;

        while (iter < max_iter &&
               residuo > tol &&
               hechos < pasos_por_llamada)
        {
            for (uint32_t j = start; j < end; j++)
                for (uint32_t i = 1; i < Nx - 1; i++)
                {
                    uint32_t idx = j * Nx + i;
                    u_new[idx] = stencil(u, i, j, Nx, r);
                }

            //barrera para que nadie calcule el residuo hasta que todos terminen su franja
            #pragma omp barrier

            double suma_local = 0.0;
            for (uint32_t j = start; j < end; j++)
                for (uint32_t i = 1; i < Nx - 1; i++)
                {
                    uint32_t idx = j * Nx + i;
                    double d = u_new[idx] - u[idx];
                    suma_local += d * d;
                }

            //reduccion del residuo local de cada hilo hacia la variable comparida
            #pragma omp atomic
            suma_global += suma_local;

            //barrera para esperar a que todos los hilos hayan sumado su parte
            #pragma omp barrier

            //solo un hilo hace el cierre 
            #pragma omp single
            {
                residuo = sqrt(suma_global / (Nx * Ny));
                suma_global = 0.0;

                for (uint32_t i = 0; i < Nx; i++)
                    u_new[i] = 100.0;

                u.swap(u_new);
                iter++;
                hechos++;
            }

            //gracias al single que trae una barrera implicita ningun hilo entra a la siguiente iteracion con informacion vieja
        }
    }
}

void ecuacion_calor_openmp_regiones(
    uint32_t Nx, uint32_t Ny,
    double Lx, double Ly,
    double alpha, double dt,
    uint32_t max_iter, double tol,
    uint32_t pasos_a_ejecutar,
    uint32_t *pixel_buffer,
    uint32_t *iter_out,
    double *residuo_out,
    double *mflops_out)
{
    static std::vector<double> u, u_new;
    static uint32_t iter_actual = 0;
    static double residuo_actual = 1e9;
    static double r = 0.0;
    static bool inicializado = false;
    static bool estable = true;  // bandera de estabilidad 

    if (!inicializado || u.size() != Nx * Ny)
    {
        u.assign(Nx * Ny, 0.0);
        u_new.assign(Nx * Ny, 0.0);
        for (uint32_t i = 0; i < Nx; ++i) { u[i] = 100.0; u_new[i] = 100.0; }

        double hx = Lx / Nx, hy = Ly / Ny;
        r = alpha * dt / (hx * hy);

        //chequeo de la condicion CFL: r debe ser <= 0.25
        estable = (r <= 0.25);

        iter_actual = 0;
        residuo_actual = 1e9;
        inicializado = true;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t iter_antes = iter_actual;

    // Solo avanzamos el solver si la condicion CFL se cumple
    if (estable)
    {
        calor_acotado_openmp(u, u_new, Nx, Ny, r, tol, iter_actual, residuo_actual,
                              pasos_a_ejecutar, max_iter);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();

    uint32_t pasos_hechos = iter_actual - iter_antes;
    double flops_totales = (double)pasos_hechos * (Nx - 2) * (Ny - 2) * 7.0;
    double mflops = (segundos > 0.0) ? (flops_totales / segundos) / 1.0e6 : 0.0;

    //pntar el pixel:buffer con reparto manual entre hilos
    #pragma omp parallel
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        uint32_t delta = (Ny + thread_count - 1) / thread_count; //techo para evitar casteos
        uint32_t start = thread_id * delta;
        uint32_t end = start + delta;
        if (end > Ny) end = Ny;

        for (uint32_t j = start; j < end; j++)
            for (uint32_t i = 0; i < Nx; i++)
            {
                uint32_t idx = j * Nx + i;
                int indice = static_cast<int>((1.0 - u[idx] / 100.0) * (color_ramp.size() - 1));
                if (indice < 0) indice = 0;
                if (indice >= static_cast<int>(color_ramp.size())) indice = color_ramp.size() - 1;
                pixel_buffer[idx] = color_ramp[indice];
            }
    }

    *iter_out = iter_actual;
    // si es inestable, usamos -1.0 como valor centinela para que main.cpp lo detecte y avise
    *residuo_out = estable ? residuo_actual : -1.0;
    *mflops_out = mflops;
}

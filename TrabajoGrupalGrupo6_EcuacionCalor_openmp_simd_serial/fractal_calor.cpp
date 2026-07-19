#include "fractal_calor.h"
#include "palette.h"
#include <vector>
#include <cmath>
#include <chrono>

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

static void calor_acotado(
    std::vector<double> &u,
    std::vector<double> &u_new,
    uint32_t Nx, uint32_t Ny,
    double r, double tol,
    uint32_t &iter, double &residuo,
    uint32_t pasos_por_llamada,
    uint32_t max_iter)
{
    uint32_t hechos = 0;

    while (iter < max_iter &&
           residuo > tol &&
           hechos < pasos_por_llamada)
    {
        for (uint32_t j = 1; j < Ny - 1; j++)
            for (uint32_t i = 1; i < Nx - 1; i++)
            {
                uint32_t idx = j * Nx + i;
                u_new[idx] = stencil(u, i, j, Nx, r);
            }

        double suma = 0.0;
        for (uint32_t j = 1; j < Ny - 1; j++)
            for (uint32_t i = 1; i < Nx - 1; i++)
            {
                uint32_t idx = j * Nx + i;
                double d = u_new[idx] - u[idx];
                suma += d * d;
            }
        residuo = sqrt(suma / (Nx * Ny));

        for (uint32_t i = 0; i < Nx; i++)
            u_new[i] = 100.0;

        u.swap(u_new);
        iter++;
        hechos++;
    }
}

void ecuacionCalorSerial(
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
    static bool estable = true;   // bandera de estabilidad 

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
        calor_acotado(u, u_new, Nx, Ny, r, tol, iter_actual, residuo_actual,
                      pasos_a_ejecutar, max_iter);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();

    uint32_t pasos_hechos = iter_actual - iter_antes;
    double flops_totales = (double)pasos_hechos * (Nx - 2) * (Ny - 2) * 7.0;
    double mflops = (segundos > 0.0) ? (flops_totales / segundos) / 1.0e6 : 0.0;

    for (uint32_t j = 0; j < Ny; j++)
        for (uint32_t i = 0; i < Nx; i++)
        {
            uint32_t idx = j * Nx + i;
            int indice = static_cast<int>((1.0 - u[idx] / 100.0) * (color_ramp.size() - 1));
            if (indice < 0) indice = 0;
            if (indice >= static_cast<int>(color_ramp.size())) indice = color_ramp.size() - 1;
            pixel_buffer[idx] = color_ramp[indice];
        }

    *iter_out = iter_actual;
    // si es inestable, usamos -1.0 como valor centinela para que main.cpp lo detecte y avise
    *residuo_out = estable ? residuo_actual : -1.0;
    *mflops_out = mflops;
}
#include "ecuacion_calor.h"
#include "palette.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

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

// evitar calculos usando datos viejos generados por los vecinos de la maya
static void intercambiar_halos(
    std::vector<double> &u, uint32_t Nx, uint32_t delta,
    int rank, int nprocs)
{
    // Identificar si el proceso tiene vecinos
    bool tiene_arriba = (rank > 0);
    bool tiene_abajo = (rank < nprocs - 1);

    // evita que los ranks envien o recivan a la vez para evitar puntos muertos
    if (rank % 2 == 0)
    {
        // Envía la última fila real
        if (tiene_abajo)
            MPI_Send(&u[delta * Nx], Nx, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        // Envía la primera fila real
        if (tiene_arriba)
            MPI_Send(&u[1 * Nx], Nx, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
        // Recibe la fila real del vecino de abajo
        if (tiene_abajo)
            MPI_Recv(&u[(delta + 1) * Nx], Nx, MPI_DOUBLE, rank + 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Recibe la fila real del vecino de arriba
        if (tiene_arriba)
            MPI_Recv(&u[0 * Nx], Nx, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
        // Espera a recibir el dato del vecino de arriba
        if (tiene_arriba)
            MPI_Recv(&u[0 * Nx], Nx, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Espera a recibir el dato del vecino de abajo
        if (tiene_abajo)
            MPI_Recv(&u[(delta + 1) * Nx], Nx, MPI_DOUBLE, rank + 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // envia su primera fila hacia arriba
        if (tiene_arriba)
            MPI_Send(&u[1 * Nx], Nx, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
        // Envía su ultima fila real hacia abajo
        if (tiene_abajo)
            MPI_Send(&u[delta * Nx], Nx, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }
}

// verificar si la ecuacion converge
static void calor_acotado_mpi(
    std::vector<double> &u,
    std::vector<double> &u_new,
    uint32_t Nx, uint32_t delta,
    uint32_t Ny, double r, double tol,
    uint32_t &iter, double &residuo,
    uint32_t pasos_por_llamada,
    uint32_t max_iter,
    int rank, int nprocs,
    uint32_t row_start, uint32_t valid_rows)
{
    uint32_t j_ini = (row_start == 0) ? 2 : 1;
    uint32_t j_fin = (row_start + valid_rows == Ny) ? (valid_rows - 1) : valid_rows;

    uint32_t hechos = 0;

    while (iter < max_iter &&
           residuo > tol &&
           hechos < pasos_por_llamada)
    {
        // PUNTO A PUNTO: intercambio de halos
        intercambiar_halos(u, Nx, delta, rank, nprocs);

        for (uint32_t j = j_ini; j <= j_fin; j++)
            for (uint32_t i = 1; i < Nx - 1; i++)
            {
                uint32_t idx = j * Nx + i;
                u_new[idx] = stencil(u, i, j, Nx, r);
            }

        double suma_local = 0.0;
        for (uint32_t j = j_ini; j <= j_fin; j++)
            for (uint32_t i = 1; i < Nx - 1; i++)
            {
                uint32_t idx = j * Nx + i;
                double d = u_new[idx] - u[idx];
                suma_local += d * d;
            }

        // COLECTIVO: Allreduce -> todos necesitan el residuo para su condicion de corte
        double suma_global = 0.0;
        MPI_Allreduce(&suma_local, &suma_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        residuo = std::sqrt(suma_global / ((double)Nx * Ny));

        if (row_start == 0)
            for (uint32_t i = 0; i < Nx; i++)
                u_new[1 * Nx + i] = 100.0;

        u.swap(u_new);
        iter++;
        hechos++;
    }
}

// Funcion principal
void ecuacionCalorMPI(
    uint32_t Nx, uint32_t Ny,
    double Lx, double Ly,
    double alpha, double dt,
    uint32_t max_iter, double tol,
    uint32_t pasos_a_ejecutar,
    int rank, int nprocs,
    uint32_t delta,
    uint32_t row_start,
    uint32_t valid_rows,
    uint32_t *pixel_buffer,
    uint32_t *iter_out,
    double *residuo_out,
    double *mflops_out,
    double *mflops_total_out)
{
    static std::vector<double> u, u_new;
    static uint32_t iter_actual = 0;
    static double residuo_actual = 1e9;
    static double r = 0.0;
    static bool inicializado = false;
    static bool estable = true;

    if (!inicializado)
    {
        // tamaño uniforme para todos los ranks de delta + 2
        u.assign((size_t)Nx * (delta + 2), 0.0);
        u_new.assign((size_t)Nx * (delta + 2), 0.0);

        // Scatter del estado inicial global
        std::vector<double> estado_global;
        if (rank == 0)
        {
            estado_global.assign((size_t)Nx * delta * nprocs, 0.0);
            for (uint32_t i = 0; i < Nx; ++i)
                estado_global[0 * Nx + i] = 100.0; // <-- fila 0
        }

        std::vector<double> mi_bloque(Nx * delta, 0.0);
        MPI_Scatter(
            rank == 0 ? estado_global.data() : nullptr, Nx * delta, MPI_DOUBLE,
            mi_bloque.data(), Nx * delta, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        for (uint32_t j = 0; j < delta; j++)
            for (uint32_t i = 0; i < Nx; i++)
                u[(j + 1) * Nx + i] = mi_bloque[j * Nx + i];

        u_new = u;

        double hx = Lx / Nx, hy = Ly / Ny;
        r = alpha * dt / (hx * hy);
        estable = (r <= 0.25);

        iter_actual = 0;
        residuo_actual = 1e9;
        inicializado = true;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t iter_antes = iter_actual;

    // Solo avanzamos el solver si la condicion CFL se cumple
    if (estable && valid_rows > 0)
    {
        calor_acotado_mpi(u, u_new, Nx, delta, Ny, r, tol,
                          iter_actual, residuo_actual,
                          pasos_a_ejecutar, max_iter,
                          rank, nprocs, row_start, valid_rows);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();

    uint32_t pasos_hechos = iter_actual - iter_antes;
    double flops_locales = (double)pasos_hechos * (double)valid_rows * (Nx - 2) * 7.0;
    double mflops_local = (segundos > 0.0) ? (flops_locales / segundos) / 1.0e6 : 0.0;

    // Reduce de solo el rank 0 necesita el total para mostrarlo en pantalla
    double mflops_total = 0.0;
    MPI_Reduce(&mflops_local, &mflops_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // pintar mi porcion local
    for (uint32_t j = 0; j < delta; j++)
    {
        for (uint32_t i = 0; i < Nx; i++)
        {
            uint32_t idx_u = (j + 1) * Nx + i;
            int indice = static_cast<int>((1.0 - u[idx_u] / 100.0) * (color_ramp.size() - 1));
            if (indice < 0)
                indice = 0;
            if (indice >= static_cast<int>(color_ramp.size()))
                indice = color_ramp.size() - 1;
            pixel_buffer[j * Nx + i] = color_ramp[indice];
        }
    }

    *iter_out = iter_actual;
    *residuo_out = estable ? residuo_actual : -1.0;
    *mflops_out = mflops_local;
    *mflops_total_out = mflops_total;
}
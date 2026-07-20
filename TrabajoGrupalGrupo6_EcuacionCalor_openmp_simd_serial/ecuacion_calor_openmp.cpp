#include "ecuacion_calor_openmp.h"
#include "palette.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <immintrin.h> // para la version combinada con AVX2 

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
    uint32_t max_iter,
    int num_threads)
{
    uint32_t hechos = 0;
    // variable compartida
    double suma_global = 0.0;

// region paralela abierta afuera del while para no crear y destruir hilos en cada iteracion
#pragma omp parallel num_threads(num_threads)
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        // reparto manual de las filas internas entre los hilos
        uint32_t filas_internas = Ny - 2;
        uint32_t delta = (filas_internas + thread_count - 1) / thread_count; // techo para evitar casteos
        uint32_t start = 1 + thread_id * delta;
        uint32_t end = start + delta;
        if (end > Ny - 1)
            end = Ny - 1;
        if (start > Ny - 1)
            start = Ny - 1;

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

// barrera para que nadie calcule el residuo hasta que todos terminen su franja
#pragma omp barrier

            double suma_local = 0.0;
            for (uint32_t j = start; j < end; j++)
                for (uint32_t i = 1; i < Nx - 1; i++)
                {
                    uint32_t idx = j * Nx + i;
                    double d = u_new[idx] - u[idx];
                    suma_local += d * d;
                }

// reduccion del residuo local de cada hilo hacia la variable comparida
#pragma omp atomic
            suma_global += suma_local;

// barrera para esperar a que todos los hilos hayan sumado su parte
#pragma omp barrier

// solo un hilo hace el cierre
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

            // gracias al single que trae una barrera implicita ningun hilo entra a la siguiente iteracion con informacion vieja
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
    double *mflops_out,
    uint32_t *threads_out,
    int num_threads)
{
    static std::vector<double> u, u_new;
    static uint32_t iter_actual = 0;
    static double residuo_actual = 1e9;
    static double r = 0.0;
    static bool inicializado = false;
    static bool estable = true; // bandera de estabilidad

    if (!inicializado || u.size() != Nx * Ny)
    {
        u.assign(Nx * Ny, 0.0);
        u_new.assign(Nx * Ny, 0.0);
        for (uint32_t i = 0; i < Nx; ++i)
        {
            u[i] = 100.0;
            u_new[i] = 100.0;
        }

        double hx = Lx / Nx, hy = Ly / Ny;
        r = alpha * dt / (hx * hy);

        // chequeo de la condicion CFL: r debe ser <= 0.25
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
                             pasos_a_ejecutar, max_iter,num_threads);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();
    uint32_t thread_count_observed = 0;

    uint32_t pasos_hechos = iter_actual - iter_antes;
    double flops_totales = (double)pasos_hechos * (Nx - 2) * (Ny - 2) * 7.0;
    double mflops = (segundos > 0.0) ? (flops_totales / segundos) / 1.0e6 : 0.0;

// pntar el pixel:buffer con reparto manual entre hilos
#pragma omp parallel num_threads(num_threads)
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

#pragma omp single
        {
            thread_count_observed = static_cast<uint32_t>(thread_count);
        }

        uint32_t delta = (Ny + thread_count - 1) / thread_count; // techo para evitar casteos
        uint32_t start = thread_id * delta;
        uint32_t end = start + delta;
        if (end > Ny)
            end = Ny;

        for (uint32_t j = start; j < end; j++)
            for (uint32_t i = 0; i < Nx; i++)
            {
                uint32_t idx = j * Nx + i;
                int indice = static_cast<int>((1.0 - u[idx] / 100.0) * (color_ramp.size() - 1));
                if (indice < 0)
                    indice = 0;
                if (indice >= static_cast<int>(color_ramp.size()))
                    indice = color_ramp.size() - 1;
                pixel_buffer[idx] = color_ramp[indice];
            }
    }

    *iter_out = iter_actual;
    // si es inestable, usamos -1.0 como valor centinela para que main.cpp lo detecte y avise
    *residuo_out = estable ? residuo_actual : -1.0;
    *mflops_out = mflops;
    if (threads_out)
        *threads_out = thread_count_observed;
}

//misma logica que calor_acotado_openmp pero el stencil y el residuo de la franja
//de cada hilo se calculan con AVX2 (4 doubles por vuelta) en vez de escalar
static void calor_acotado_openmp_simd(
    std::vector<double> &u,
    std::vector<double> &u_new,
    uint32_t Nx, uint32_t Ny,
    double r, double tol,
    uint32_t &iter, double &residuo,
    uint32_t pasos_por_llamada,
    uint32_t max_iter,
    int num_threads)
{
    uint32_t hechos = 0;
    //variable compartida
    double suma_global = 0.0;

//region paralela abierta afuera del while para no crear y destruir hilos en cada iteracion
#pragma omp parallel num_threads(num_threads)
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        //reparto manual de las filas internas entre los hilos, igual que en la version escalar
        uint32_t filas_internas = Ny - 2;
        uint32_t delta = (filas_internas + thread_count - 1) / thread_count; //techo para evitar casteos
        uint32_t start = 1 + thread_id * delta;
        uint32_t end = start + delta;
        if (end > Ny - 1)
            end = Ny - 1;
        if (start > Ny - 1)
            start = Ny - 1;

        //constantes vectoriales, se calculan una sola vez por hilo ya que r no cambia
        __m256d coef_r = _mm256_set1_pd(r);
        __m256d cuatro = _mm256_set1_pd(4.0);

        while (iter < max_iter &&
               residuo > tol &&
               hechos < pasos_por_llamada)
        {
            //stencil de la franja del hilo, 4 doubles por vuelta + cola escalar
            for (uint32_t j = start; j < end; j++)
            {
                uint32_t base = j * Nx;
                uint32_t base_arriba = (j - 1) * Nx;
                uint32_t base_abajo = (j + 1) * Nx;

                uint32_t i = 1;
                for (; i + 3 < Nx - 1; i += 4)
                {
                    __m256d centro    = _mm256_loadu_pd(&u[base + i]);
                    __m256d izquierda = _mm256_loadu_pd(&u[base + i - 1]);
                    __m256d derecha   = _mm256_loadu_pd(&u[base + i + 1]);
                    __m256d arriba    = _mm256_loadu_pd(&u[base_arriba + i]);
                    __m256d abajo     = _mm256_loadu_pd(&u[base_abajo + i]);

                    __m256d suma_vecinos = _mm256_add_pd(izquierda, derecha);
                    suma_vecinos = _mm256_add_pd(suma_vecinos, arriba);
                    suma_vecinos = _mm256_add_pd(suma_vecinos, abajo);

                    __m256d termino = _mm256_sub_pd(suma_vecinos, _mm256_mul_pd(cuatro, centro));
                    __m256d nuevo = _mm256_add_pd(centro, _mm256_mul_pd(coef_r, termino));

                    _mm256_storeu_pd(&u_new[base + i], nuevo);
                }

                //resto escalar de la fila que no completa un bloque de 4
                for (; i < Nx - 1; ++i)
                {
                    uint32_t idx = base + i;
                    u_new[idx] = u[idx] +
                        r * (u[idx + 1] + u[idx - 1] +
                             u[base_arriba + i] + u[base_abajo + i] -
                             4.0 * u[idx]);
                }
            }

//barrera para que nadie calcule el residuo hasta que todos terminen su franja
#pragma omp barrier

            // residuo local del hilo: reduccion SIMD por bloques de 4 + cola escalar
            __m256d suma_cuadrados = _mm256_setzero_pd();
            double suma_escalar = 0.0;

            for (uint32_t j = start; j < end; j++)
            {
                uint32_t base = j * Nx;
                uint32_t i = 1;
                for (; i + 3 < Nx - 1; i += 4)
                {
                    __m256d nuevo = _mm256_loadu_pd(&u_new[base + i]);
                    __m256d viejo = _mm256_loadu_pd(&u[base + i]);
                    __m256d diff  = _mm256_sub_pd(nuevo, viejo);
                    __m256d diff2 = _mm256_mul_pd(diff, diff);
                    suma_cuadrados = _mm256_add_pd(suma_cuadrados, diff2);
                }
                for (; i < Nx - 1; ++i)
                {
                    double d = u_new[base + i] - u[base + i];
                    suma_escalar += d * d;
                }
            }

            //reduccion horizontal del vector de 4 doubles a un solo double
            __m128d low    = _mm256_castpd256_pd128(suma_cuadrados);
            __m128d high   = _mm256_extractf128_pd(suma_cuadrados, 1);
            __m128d sum128 = _mm_add_pd(low, high);
            __m128d sum64  = _mm_hadd_pd(sum128, sum128);
            double suma_local = _mm_cvtsd_f64(sum64) + suma_escalar;

//reduccion del residuo local de cada hilo hacia la variable compartida
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

            //misma razon que en la version escalar: el single evita datos viejos en la siguiente vuelta
        }
    }
}

//misma funcion publica que ecuacion_calor_openmp_regiones pero delegando en el
//solver vectorizado con AVX2 (calor_acotado_openmp_simd)
void ecuacion_calor_openmp_simd(
    uint32_t Nx, uint32_t Ny,
    double Lx, double Ly,
    double alpha, double dt,
    uint32_t max_iter, double tol,
    uint32_t pasos_a_ejecutar,
    uint32_t *pixel_buffer,
    uint32_t *iter_out,
    double *residuo_out,
    double *mflops_out,
    uint32_t *threads_out,
    int num_threads)
{
    static std::vector<double> u, u_new;
    static uint32_t iter_actual = 0;
    static double residuo_actual = 1e9;
    static double r = 0.0;
    static bool inicializado = false;
    static bool estable = true; //bandera de estabilidad

    if (!inicializado || u.size() != Nx * Ny)
    {
        u.assign(Nx * Ny, 0.0);
        u_new.assign(Nx * Ny, 0.0);
        for (uint32_t i = 0; i < Nx; ++i)
        {
            u[i] = 100.0;
            u_new[i] = 100.0;
        }

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

    //Solo avanzamos el solver si la condicion CFL se cumple
    if (estable)
    {
        calor_acotado_openmp_simd(u, u_new, Nx, Ny, r, tol, iter_actual, residuo_actual,
                                  pasos_a_ejecutar, max_iter, num_threads);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();
    uint32_t thread_count_observed = 0;

    uint32_t pasos_hechos = iter_actual - iter_antes;
    double flops_totales = (double)pasos_hechos * (Nx - 2) * (Ny - 2) * 7.0;
    double mflops = (segundos > 0.0) ? (flops_totales / segundos) / 1.0e6 : 0.0;

//pintar el pixel_buffer con reparto manual entre hilos, igual que la version escalar
#pragma omp parallel num_threads(num_threads)
    {
        int thread_count = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

#pragma omp single
        {
            thread_count_observed = static_cast<uint32_t>(thread_count);
        }

        uint32_t delta = (Ny + thread_count - 1) / thread_count; //techo para evitar casteos
        uint32_t start = thread_id * delta;
        uint32_t end = start + delta;
        if (end > Ny)
            end = Ny;

        for (uint32_t j = start; j < end; j++)
            for (uint32_t i = 0; i < Nx; i++)
            {
                uint32_t idx = j * Nx + i;
                int indice = static_cast<int>((1.0 - u[idx] / 100.0) * (color_ramp.size() - 1));
                if (indice < 0)
                    indice = 0;
                if (indice >= static_cast<int>(color_ramp.size()))
                    indice = color_ramp.size() - 1;
                pixel_buffer[idx] = color_ramp[indice];
            }
    }

    *iter_out = iter_actual;
    //si es inestable, usamos -1.0 como valor centinela para que main.cpp lo detecte y avise
    *residuo_out = estable ? residuo_actual : -1.0;
    *mflops_out = mflops;
    if (threads_out)
        *threads_out = thread_count_observed;
}


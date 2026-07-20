#include "fractal_calor_simd.h"
#include "palette.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <immintrin.h>   // AVX2

static void paso_calor_simd(
    std::vector<double> &campo,
    std::vector<double> &campo_nuevo,
    uint32_t Nx, uint32_t Ny,
    double r,
    double &residuo)
{

    __m256d coef_r = _mm256_set1_pd(r);
    __m256d cuatro = _mm256_set1_pd(4.0);

    for (uint32_t j = 1; j < Ny - 1; ++j)
    {
        uint32_t base = j * Nx;
        uint32_t base_arriba = (j - 1) * Nx;
        uint32_t base_abajo = (j + 1) * Nx;

        uint32_t i = 1;
        for (; i + 3 < Nx - 1; i += 4)
        {
            __m256d centro   = _mm256_loadu_pd(&campo[base + i]);
            __m256d izquierda = _mm256_loadu_pd(&campo[base + i - 1]);
            __m256d derecha   = _mm256_loadu_pd(&campo[base + i + 1]);
            __m256d arriba    = _mm256_loadu_pd(&campo[base_arriba + i]);
            __m256d abajo     = _mm256_loadu_pd(&campo[base_abajo + i]);

            __m256d suma_vecinos = _mm256_add_pd(izquierda, derecha);
            suma_vecinos = _mm256_add_pd(suma_vecinos, arriba);
            suma_vecinos = _mm256_add_pd(suma_vecinos, abajo);

            __m256d termino = _mm256_sub_pd(suma_vecinos, _mm256_mul_pd(cuatro, centro));
            __m256d nuevo = _mm256_add_pd(centro, _mm256_mul_pd(coef_r, termino));

            _mm256_storeu_pd(&campo_nuevo[base + i], nuevo);
        }

        // Resto escalar
        for (; i < Nx - 1; ++i)
        {
            uint32_t idx = base + i;
            campo_nuevo[idx] = campo[idx] +
                r * (campo[idx+1] + campo[idx-1] +
                    campo[base_arriba + i] + campo[base_abajo + i] -
                     4.0 * campo[idx]);
        }
    }

    __m256d suma_cuadrados = _mm256_setzero_pd();
    double suma_escalar = 0.0;

    for (uint32_t j = 1; j < Ny - 1; ++j)
    {
        uint32_t base = j * Nx;
        uint32_t i = 1;
        for (; i + 3 < Nx - 1; i += 4)
        {
            __m256d nuevo = _mm256_loadu_pd(&campo_nuevo[base + i]);
            __m256d viejo = _mm256_loadu_pd(&campo[base + i]);
            __m256d diff  = _mm256_sub_pd(nuevo, viejo);
            __m256d diff2 = _mm256_mul_pd(diff, diff);
            suma_cuadrados = _mm256_add_pd(suma_cuadrados, diff2);
        }
        for (; i < Nx - 1; ++i)
        {
            double d = campo_nuevo[base + i] - campo[base + i];
            suma_escalar += d * d;
        }
    }

    // reduccion del vector de 4 doubles
    __m128d low  = _mm256_castpd256_pd128(suma_cuadrados);
    __m128d high = _mm256_extractf128_pd(suma_cuadrados, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    __m128d sum64  = _mm_hadd_pd(sum128, sum128);
    double suma_simd = _mm_cvtsd_f64(sum64);

    double suma_total = suma_simd + suma_escalar;
    residuo = sqrt(suma_total / (Nx * Ny));

    // frontera superior
    for (uint32_t i = 0; i < Nx; ++i)
        campo_nuevo[i] = 100.0;

    // intercambiar punteros
    campo.swap(campo_nuevo);
}

void ecuacion_Calor_SIMD(
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
    static std::vector<double> campo, campo_nuevo;
    static uint32_t iter_actual = 0;
    static double residuo_actual = 1e9;
    static double r = 0.0;
    static bool inicializado = false;
    static bool estable = true;

    // redimensionar si cambia el tamaño
    if (!inicializado || campo.size() != Nx * Ny)
    {
        campo.assign(Nx * Ny, 0.0);
        campo_nuevo.assign(Nx * Ny, 0.0);
        for (uint32_t i = 0; i < Nx; ++i) {
            campo[i] = 100.0;
            campo_nuevo[i] = 100.0;
        }

        double hx = Lx / Nx, hy = Ly / Ny;
        r = alpha * dt / (hx * hy);
        estable = (r <= 0.25);

        iter_actual = 0;
        residuo_actual = 1e9;
        inicializado = true;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t iter_antes = iter_actual;

    // estable avanza
    if (estable)
    {
        uint32_t hechos = 0;
        while (iter_actual < max_iter &&
            residuo_actual > tol &&
            hechos < pasos_a_ejecutar)
        {
            paso_calor_simd(campo, campo_nuevo, Nx, Ny, r, residuo_actual);
            iter_actual++;
            hechos++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double segundos = std::chrono::duration<double>(t1 - t0).count();

    uint32_t pasos_hechos = iter_actual - iter_antes;
    double flops_totales = (double)pasos_hechos * (Nx - 2) * (Ny - 2) * 7.0;
    double mflops = (segundos > 0.0) ? (flops_totales / segundos) / 1.0e6 : 0.0;

    //buffer de píxeles
    for (uint32_t j = 0; j < Ny; ++j)
        for (uint32_t i = 0; i < Nx; ++i)
        {
            uint32_t idx = j * Nx + i;
            int indice = static_cast<int>((1.0 - campo[idx] / 100.0) * (color_ramp.size() - 1));
            if (indice < 0) indice = 0;
            if (indice >= static_cast<int>(color_ramp.size())) indice = color_ramp.size() - 1;
            pixel_buffer[idx] = color_ramp[indice];
        }

    *iter_out = iter_actual;
    *residuo_out = estable ? residuo_actual : -1.0;
    *mflops_out = mflops;
}
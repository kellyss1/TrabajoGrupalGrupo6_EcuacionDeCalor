#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/core.h>
#include <SFML/Graphics.hpp>
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include "ecuacion_calor.h"
#include "palette.h"

#define ANCHO 1600
#define ALTO 900

uint32_t Nx = 1024;
uint32_t Ny = 1024;
double Lx = 1.0, Ly = 1.0, alpha = 0.25, dt = 5.0e-7, tol = 1.0e-4;
int max_iteraciones = 10000;
bool modo_continuo = false;
bool solicitar_paso = false;
int running = 1;

uint32_t *pixel_buffer = nullptr;  // tamaño Nx*delta, porcion local de los ranks diferentes a 0
uint32_t *campo_colores = nullptr; // solo para el rank 0

uint32_t delta, row_start, valid_rows;
int nprocs, rank;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // divicion de trabajo de los ranks usando el ceil de ny / procesos
    delta = (uint32_t)std::ceil(1.0 * Ny / nprocs);
    row_start = rank * delta;
    uint32_t row_end = row_start + delta;
    if (row_end > Ny)
        row_end = Ny;
    valid_rows = (row_end > row_start) ? (row_end - row_start) : 0;

    pixel_buffer = new uint32_t[Nx * delta];
    std::memset(pixel_buffer, 0, Nx * delta * sizeof(uint32_t));

    fmt::print("Rank {}: rows {} to {} (validas: {})\n", rank, row_start, row_end, valid_rows);

    if (rank == 0)
    {
        campo_colores = new uint32_t[Nx * delta * nprocs];
        std::memset(campo_colores, 0, Nx * delta * nprocs * sizeof(uint32_t));

        sf::RenderWindow window(sf::VideoMode({ANCHO, ALTO}), "Mapa de Calor MPI");
#ifdef _WIN32
        HWND hwnd = window.getNativeHandle();
        ShowWindow(hwnd, SW_MAXIMIZE);
#endif
        sf::Texture texture({ANCHO, ALTO});
        sf::Sprite sprite(texture);

        sf::Font font("arial.ttf");
        sf::Text text(font, "Mapa de Calor MPI", 24);
        text.setFillColor(sf::Color::White);
        text.setPosition({10, 10});
        text.setStyle(sf::Text::Bold);

        std::string options = "Opciones: [1] Continuo/Pausa | [2] Paso a paso";
        sf::Text textoptions(font, options, 20);
        textoptions.setStyle(sf::Text::Bold);
        textoptions.setPosition({10, window.getView().getSize().y - 40});

        int frames = 0, fps = 0;
        sf::Clock clock;
        uint32_t iter_ui = 0;
        double residuo_ui = 1e9, mflops_ui = 0.0, mflops_total_ui = 0.0;
        double tiempo_ejecucion_ui = 0.0;
        bool acumulando_tiempo_continuo = false;

        auto format_time_hms = [](double milisegundos)
        {
            int total_milisegundos = static_cast<int>(milisegundos);
            int minutos = total_milisegundos / 60000;
            int segundos = (total_milisegundos % 60000) / 1000;
            int ms = total_milisegundos % 1000;
            return fmt::format("{:02d}:{:02d}:{:03d}", minutos, segundos, ms);
        };

        while (window.isOpen())
        {
            while (const std::optional event = window.pollEvent())
            {
                if (event->is<sf::Event::Closed>())
                {
                    running = 0;
                    window.close();
                }
                else if (event->is<sf::Event::KeyReleased>())
                {
                    auto evt = event->getIf<sf::Event::KeyReleased>();
                    switch (evt->scancode)
                    {
                    case sf::Keyboard::Scan::Num1:
                        modo_continuo = !modo_continuo;
                        break;
                    case sf::Keyboard::Scan::Num2:
                        if (!modo_continuo)
                            solicitar_paso = true;
                        break;
                    }
                }
            }

            bool solver_terminado = iter_ui >= static_cast<uint32_t>(max_iteraciones);

            uint32_t pasos = 0;
            if (!solver_terminado)
            {
                if (modo_continuo)
                    pasos = 1;
                else if (solicitar_paso)
                {
                    pasos = 20;
                    solicitar_paso = false;
                }
            }

            // Bcast de controles si esta ejecutandose y el pasos de las iteraciones maximas para el modo continuo, asi como el tope de las iteraciones
            std::vector<int> ctrl = {running, (int)pasos, max_iteraciones};
            MPI_Bcast(ctrl.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

            if (running == 0)
                break;

            if (modo_continuo && !solver_terminado)
            {
                if (!acumulando_tiempo_continuo)
                {
                    tiempo_ejecucion_ui = 0.0;
                    acumulando_tiempo_continuo = true;
                }

                sf::Clock solver_clock;
                ecuacionCalorMPI(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                 pasos, rank, nprocs, delta, row_start, valid_rows,
                                 pixel_buffer, &iter_ui, &residuo_ui, &mflops_ui, &mflops_total_ui);
                tiempo_ejecucion_ui += solver_clock.getElapsedTime().asSeconds() * 1000.0;
            }
            else if (!solver_terminado)
            {
                sf::Clock solver_clock;
                ecuacionCalorMPI(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                 pasos, rank, nprocs, delta, row_start, valid_rows,
                                 pixel_buffer, &iter_ui, &residuo_ui, &mflops_ui, &mflops_total_ui);
                tiempo_ejecucion_ui = solver_clock.getElapsedTime().asSeconds() * 1000.0;
            }
            else
            {
                ecuacionCalorMPI(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                 pasos, rank, nprocs, delta, row_start, valid_rows,
                                 pixel_buffer, &iter_ui, &residuo_ui, &mflops_ui, &mflops_total_ui);
            }

            // Gather que junta la porcion de todos los ranks
            MPI_Gather(
                pixel_buffer, Nx * delta, MPI_UNSIGNED,
                campo_colores, Nx * delta, MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

            // pintar los colores en pantalla escalando a los valores reales de pantalla
            static std::vector<uint32_t> pixel_ventana(ANCHO * ALTO);
            for (uint32_t py = 0; py < ALTO; ++py)
            {
                uint32_t j = (uint32_t)(((uint64_t)py * Ny) / ALTO);
                if (j >= Ny)
                    j = Ny - 1;
                for (uint32_t px = 0; px < ANCHO; ++px)
                {
                    uint32_t i = (uint32_t)(((uint64_t)px * Nx) / ANCHO);
                    if (i >= Nx)
                        i = Nx - 1;
                    pixel_ventana[py * ANCHO + px] = campo_colores[j * Nx + i];
                }
            }
            texture.update((const uint8_t *)pixel_ventana.data());

            frames++;
            if (clock.getElapsedTime().asSeconds() >= 1.0f)
            {
                fps = frames;
                frames = 0;
                clock.restart();
            }

            auto msg = fmt::format(
                "Backend: MPI ({} procs) | Iter: {}/{} | Residuo L2: {:.6e} | MFLOPS total: {:.1f} | Modo: {} | FPS: {}{}",
                nprocs, iter_ui, max_iteraciones, residuo_ui, mflops_total_ui,
                modo_continuo ? "CONTINUO" : "PASO A PASO", fps,
                modo_continuo ? fmt::format("\nTiempo total de ejecucion: {}", format_time_hms(tiempo_ejecucion_ui)) : std::string(""));
            text.setString(msg);

            window.clear();
            window.draw(sprite);
            window.draw(text);
            window.draw(textoptions);
            window.display();
        }

        delete[] campo_colores;
    }
    else
    {
        while (true)
        {
            std::vector<int> ctrl(3);
            MPI_Bcast(ctrl.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
            running = ctrl[0];
            uint32_t pasos = (uint32_t)ctrl[1];
            max_iteraciones = ctrl[2];

            if (running == 0)
            {
                fmt::print("Rank {}: cerrando...\n", rank);
                break;
            }

            uint32_t iter_ui = 0;
            double residuo_ui = 1e9, mflops_ui = 0.0, mflops_total_ui = 0.0;

            ecuacionCalorMPI(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                             pasos, rank, nprocs, delta, row_start, valid_rows,
                             pixel_buffer, &iter_ui, &residuo_ui, &mflops_ui, &mflops_total_ui);

            // Gather los demas ranks solo aportan su parte y la guardan el el pixel buffer general
            MPI_Gather(
                pixel_buffer, Nx * delta, MPI_UNSIGNED,
                nullptr, 0, MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
        }
    }

    delete[] pixel_buffer;
    MPI_Finalize();
    return 0;
}
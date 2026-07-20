#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/core.h>
#include <SFML/Graphics.hpp>
#include <omp.h>
#include <cstdint>
#include <complex>
#include "fractal_calor.h"
#include "ecuacion_calor_openmp.h"
#include "fractal_calor_simd.h"

// Pamarametro img
#define ANCHO 1600
#define ALTO 900

// Parameteros
int max_iteraciones = 10000;
uint32_t Nx = 1024;
uint32_t Ny = 1024;
double Lx = 1.0;
double Ly = 1.0;
double alpha = 0.25;
double dt = 5.0e-7;
double tol = 1.0e-4;
bool modo_continuo = false;
bool solicitar_paso = false;
// Variables para mostrar en la interfaz
uint32_t iter_actual_ui = 0;
double residuo_ui = 1e9;
double mflops_ui = 0.0;
uint32_t threads_ui = 0;
double tiempo_ejecucion_ui = 0.0;
bool acumulando_tiempo_continuo = false;

// textura (uint32_t tipo de dato sin signo de 16/32/64 etc)
uint32_t *pixel_buffer = nullptr;

enum class runtime_type
{
    SERIAL_1 = 0,
    SIMD,
    OPENMP,
};

int main()
{
    // int thread_count;
    // #pragma omp parallel
    //{
    // #pragma opm master
    //{
    //  thread_count = omp_get_num_threads();
    //}
    // }
    runtime_type r_type = runtime_type::SERIAL_1;
    // en un solo vector puede representar todo la img ya sea por filas o columnas
    pixel_buffer = new uint32_t[ANCHO * ALTO];

    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();

    sf::RenderWindow window(sf::VideoMode({ANCHO, ALTO}), "Mapa de Calor");

#ifdef _WIN32
    HWND hwnd = window.getNativeHandle();
    ShowWindow(hwnd, SW_MAXIMIZE); // Maximizar Ventana
#endif

    sf::Texture texture({ANCHO, ALTO});
    sf::Sprite sprite(texture);

    sf::Font font("arial.ttf");
    sf::Text text(font, "Mapa de Calor", 24);
    text.setFillColor(sf::Color::White);
    text.setPosition({10, 10});
    text.setStyle(sf::Text::Bold);

    std::string options = "Options: [1]Serial [2]SIMD [3]OPENMP | [Space] Modo continuo/pausa | [Enter] Paso a paso";

    sf::Text textoptions(font, options, 20);

    textoptions.setStyle(sf::Text::Bold);
    textoptions.setPosition({10, window.getView().getSize().y - 40});
    // FPS
    int frames = 0;
    int fps = 0;
    sf::Clock clock;

    // Start the game loop
    while (window.isOpen())
    {
        // Process events
        while (const std::optional event = window.pollEvent())
        {
            // Close window: exit
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (event->is<sf::Event::KeyReleased>())
            {
                auto evt = event->getIf<sf::Event::KeyReleased>();

                switch (evt->scancode)
                {
                case sf::Keyboard::Scan::Up:
                    max_iteraciones += 1000;
                    break;
                case sf::Keyboard::Scan::Down:
                    max_iteraciones -= 1000;
                    if (max_iteraciones < 1000)
                        max_iteraciones = 1000;
                    break;
                case sf::Keyboard::Scan::Num1:
                    r_type = runtime_type::SERIAL_1;
                    tiempo_ejecucion_ui = 0.0;
                    acumulando_tiempo_continuo = false;
                    break;
                case sf::Keyboard::Scan::Num2:
                    r_type = runtime_type::SIMD;
                    tiempo_ejecucion_ui = 0.0;
                    acumulando_tiempo_continuo = false;
                    break;
                case sf::Keyboard::Scan::Num3:
                    r_type = runtime_type::OPENMP;
                    tiempo_ejecucion_ui = 0.0;
                    acumulando_tiempo_continuo = false;
                    break;
                case sf::Keyboard::Scan::Space:
                    modo_continuo = !modo_continuo; // alterna paso a paso / continuo
                    break;
                case sf::Keyboard::Scan::Enter:
                    if (!modo_continuo)
                        solicitar_paso = true; // un solo paso, solo si está pausado
                    break;
                }
            }
        }
        std::string mode = "";
        // Dibuja Fractal dependera de la velocidad
        if (r_type == runtime_type::SERIAL_1)
        {
            uint32_t pasos = 0;
            if (modo_continuo)
                pasos = 1; // 1 paso por frame en modo continuo
            else if (solicitar_paso)
            {
                pasos = 20; // 20 pasos, una sola vez
                solicitar_paso = false;
            }

            bool solver_terminado = iter_actual_ui >= static_cast<uint32_t>(max_iteraciones);
            if (modo_continuo && !solver_terminado)
            {
                if (!acumulando_tiempo_continuo)
                {
                    tiempo_ejecucion_ui = 0.0;
                    acumulando_tiempo_continuo = true;
                }

                sf::Clock solver_clock;
                ecuacionCalorSerial(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                    pasos, pixel_buffer,
                                    &iter_actual_ui, &residuo_ui, &mflops_ui);
                tiempo_ejecucion_ui += solver_clock.getElapsedTime().asSeconds() * 1000.0;
            }
            else if (!solver_terminado)
            {
                acumulando_tiempo_continuo = false;
                sf::Clock solver_clock;
                ecuacionCalorSerial(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                    pasos, pixel_buffer,
                                    &iter_actual_ui, &residuo_ui, &mflops_ui);
                tiempo_ejecucion_ui = solver_clock.getElapsedTime().asSeconds() * 1000.0;
            }
            threads_ui = 0;
            mode = "SERIAL 1";
        }
        else if (r_type == runtime_type::SIMD)
        {
            uint32_t pasos = 0;
            if (modo_continuo)
                pasos = 1; // 1 paso por frame en modo continuo
            else if (solicitar_paso)
            {
                pasos = 20; // 20 pasos, una sola vez
                solicitar_paso = false;
            }

            bool solver_terminado = iter_actual_ui >= static_cast<uint32_t>(max_iteraciones);
            if (modo_continuo && !solver_terminado)
            {
                if (!acumulando_tiempo_continuo)
                {
                    tiempo_ejecucion_ui = 0.0;
                    acumulando_tiempo_continuo = true;
                }

                sf::Clock solver_clock;
                ecuacion_Calor_SIMD(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                    pasos, pixel_buffer,
                                    &iter_actual_ui, &residuo_ui, &mflops_ui);
                tiempo_ejecucion_ui += solver_clock.getElapsedTime().asSeconds() * 1000.0;
            }
            else if (!solver_terminado)
            {
                acumulando_tiempo_continuo = false;
                sf::Clock solver_clock;
                ecuacion_Calor_SIMD(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                    pasos, pixel_buffer,
                                    &iter_actual_ui, &residuo_ui, &mflops_ui);
                tiempo_ejecucion_ui = solver_clock.getElapsedTime().asSeconds() * 1000.0;
            }
            threads_ui = 0;
            mode = "SIMD";
        }
        else if (r_type == runtime_type::OPENMP)
        {
            uint32_t pasos = 0;
            if (modo_continuo)
                pasos = 1; // 1 paso por frame en modo continuo
            else if (solicitar_paso)
            {
                pasos = 20; // 20 pasos, una sola vez
                solicitar_paso = false;
            }

            bool solver_terminado = iter_actual_ui >= static_cast<uint32_t>(max_iteraciones);

            if (modo_continuo && !solver_terminado)
            {
                if (!acumulando_tiempo_continuo)
                {
                    tiempo_ejecucion_ui = 0.0;
                    acumulando_tiempo_continuo = true;
                }

                sf::Clock solver_clock;

                ecuacion_calor_openmp_regiones(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                    pasos, pixel_buffer,
                                    &iter_actual_ui, &residuo_ui, &mflops_ui, &threads_ui);
                tiempo_ejecucion_ui += solver_clock.getElapsedTime().asSeconds() * 1000.0;
            }
            else
            {
                if (!solver_terminado)
                {
                    acumulando_tiempo_continuo = false;
                    tiempo_ejecucion_ui = 0.0;

                    sf::Clock solver_clock;
                    ecuacion_calor_openmp_regiones(Nx, Ny, Lx, Ly, alpha, dt, (uint32_t)max_iteraciones, tol,
                                        pasos, pixel_buffer,
                                        &iter_actual_ui, &residuo_ui, &mflops_ui, &threads_ui);
                    if (iter_actual_ui < static_cast<uint32_t>(max_iteraciones))
                        tiempo_ejecucion_ui = solver_clock.getElapsedTime().asSeconds() * 1000.0;
                }
            }
            mode = "OPENMP";
        }

        texture.update((const uint8_t *)pixel_buffer);

        frames++;

        if (clock.getElapsedTime().asSeconds() >= 1.0f)
        {
            fps = frames;
            frames = 0;
            clock.restart();
        }
        // overlay esquina superior izquierda
        std::string estado_cfl = (residuo_ui < 0.0) ? "INESTABLE (r > 0.25, ajusta dt/Nx/alpha)" : "OK";

        auto format_time_hms = [](double milisegundos)
        {
            int total_milisegundos = static_cast<int>(milisegundos);
            int minutos = total_milisegundos / 60000;
            int segundos = (total_milisegundos % 60000) / 1000;
            int ms = total_milisegundos % 1000;
            return fmt::format("{:02d}:{:02d}:{:03d}", minutos, segundos, ms);
        };

        std::string extra_info = "";
        if (modo_continuo)
            extra_info = fmt::format("\nTiempo total de ejecucion: {}", format_time_hms(tiempo_ejecucion_ui));

        if (r_type == runtime_type::OPENMP)
        {
            extra_info = fmt::format("\nHilos OpenMP: {}{}",
                                     threads_ui,
                                     modo_continuo ? fmt::format("\nTiempo total de ejecucion: {}", format_time_hms(tiempo_ejecucion_ui)) : "");
        }

        auto msg = fmt::format(
            "Backend: {} | Iter: {}/{} | Residuo L2: {} | MFLOPS: {:.1f} | Modo: {} | FPS: {} | CFL: {}{}",
            mode, iter_actual_ui,
            max_iteraciones,
            (residuo_ui < 0.0) ? std::string("N/A") : fmt::format("{:.6e}", residuo_ui),
            mflops_ui,
            modo_continuo ? "CONTINUO" : "PASO A PASO", fps,
            estado_cfl,
            extra_info);
        text.setString(msg);
        auto bounds = text.getLocalBounds();
        text.setOrigin({bounds.position.x + bounds.size.x / 2.0f, bounds.position.y + bounds.size.y / 2.0f});
        text.setPosition({window.getView().getSize().x / 2.0f, window.getView().getSize().y / 2.0f});
        // Clear screen
        window.clear();
        {
            window.draw(sprite);
            window.draw(text);
            window.draw(textoptions);
        }

        // Update the window
        window.display();
    }

    delete[] pixel_buffer;
    return 0;
}

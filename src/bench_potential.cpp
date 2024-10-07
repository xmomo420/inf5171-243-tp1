#include <tbb/global_control.h>

#include <chrono>

#include "colormap.h"
#include "particle.h"
#include "potentialparallel.h"
#include "uqam/tp.h"

int main(int argc, char **argv) {

  // Paramètres (à ajuster pour votre situation)
  int resolution = 500;
  std::vector<Particle> particles;
  double dt = 1e-9;
  int substeps = 10;
  int numpart = 100;
  int max_iter = 200;  // Nombre d'itérations pour la simulation
  int repeat = 10;
  ColorMap cmap;
  cmap.load(SOURCE_DIR "/data/colormap_parula.png");

  // Généreration de 100 particules aléatoires
  for (int i = 0; i < numpart; ++i) {
    particles.emplace_back(Vector2d::Random(), Vector2d::Zero(), 1e-6);
  }

  std::ofstream ofs("bench_potential.dat");
  ofs << "# ncpu temps acceleration" << std::endl;

  // SÉRIE
  long serial_time_ns = 0.0;
  for (int r = 0; r < repeat; ++r) {
    PotentialSerial potential_serial(resolution, resolution);
    auto start = std::chrono::high_resolution_clock::now();

    potential_serial.run(particles, max_iter, dt, substeps, true, cmap, "serial", false);

    auto end = std::chrono::high_resolution_clock::now();
    serial_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  serial_time_ns /= repeat;  // Moyenne du temps série

  // PARALLEL
  unsigned int max_threads = std::thread::hardware_concurrency();  // Détection du nombre de processeurs
  for (int ncpu = 1; ncpu <= max_threads; ++ncpu) {
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, ncpu);

    double parallel_time_ns = 0.0;
    for (int r = 0; r < repeat; ++r) {
      PotentialParallel potential_parallel(resolution, resolution);
      auto start = std::chrono::high_resolution_clock::now();

      potential_parallel.run(particles, max_iter, dt, substeps, true, cmap, "parallel", false);

      auto end = std::chrono::high_resolution_clock::now();
      parallel_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    parallel_time_ns /= repeat;  // Moyenne du temps parallèle

    double acceleration = static_cast<double>(serial_time_ns) / parallel_time_ns;  // Calcul de l'accélération

    ofs << ncpu << " " << parallel_time_ns << " " << acceleration << std::endl;
  }

  ofs.close();
  std::cout << "DONE!" << std::endl;
  return 0;

}

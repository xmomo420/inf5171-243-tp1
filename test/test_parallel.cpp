#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <colormap.h>
#include <particle.h>
#include <potential.h>
#include <potentialparallel.h>
#include <uqam/tp.h>

#define DATA_DIR SOURCE_DIR "/test/data"

using namespace Catch;

static const double abstol = 1e-6;

TEST_CASE("PotentialParallel") {
  int width = 512;
  int height = 512;
  std::vector particles = {
    Particle({0.25, 0.25}, {0, 0}, 1e-6),
    Particle({0.75, 0.75}, {0, 0}, -1e-6),
  };

  PotentialSerial serial(width, height);
  PotentialParallel parallel(width, height);

  SECTION("ComputeField") {
    double hi_parallel;
    double lo_parallel;
    double hi_serial;
    double lo_serial;
    serial.compute_field(particles, lo_serial, hi_serial);
    parallel.compute_field(particles, lo_parallel, hi_parallel);

    // Comparer chaque élément de m_sol avec une tolérance
    REQUIRE(serial.m_sol.size() == parallel.m_sol.size());
    for (size_t i = 0; i < serial.m_sol.size(); ++i) {
      REQUIRE_THAT(serial.m_sol[i], Matchers::WithinAbs(parallel.m_sol[i], abstol));
    }

    REQUIRE_THAT(lo_serial, Matchers::WithinAbs(lo_parallel, abstol));
    REQUIRE_THAT(hi_serial, Matchers::WithinAbs(hi_parallel, abstol));
  }

  SECTION("MoveParticles") {
    int substeps = 10;
    double dt = 1e-9;
    serial.move_particles(particles, dt, substeps);
    parallel.move_particles(particles, dt, substeps);

    for (size_t i = 0; i < particles.size(); ++i) {
      REQUIRE_THAT(serial.m_sol[i], Matchers::WithinAbs(parallel.m_sol[i], abstol));
    }
  }

  SECTION("SaveSolution") {
    ColorMap cmap;
    std::ostringstream oss_serial, oss_parallel;

    serial.save_solution(oss_serial, cmap);
    parallel.save_solution(oss_parallel, cmap);

    REQUIRE(oss_serial.str() == oss_parallel.str());
  }

}
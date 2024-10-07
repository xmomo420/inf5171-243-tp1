#include "potentialparallel.h"
#include <tbb/spin_mutex.h>

#include <oneapi/tbb/parallel_for.h>

/*
 * IDENTIFICATION :
 * BENM23109703
 */

tbb::spin_mutex mutex; // Variable globale Mutex pour protéger l'accès à lo et hi

void PotentialParallel::compute_field(std::vector<Particle>& charge, double& lo, double& hi) {
  // TODO : Parallisation de la fonction originale dans potential.cpp
  lo = std::numeric_limits<double>::max();
  hi = std::numeric_limits<double>::min();
  const int n = static_cast<int>(charge.size());

  // Tableau pour stocker les potentiels calculés
  std::vector potentials(m_height * m_width, 0.0);

  // Paralleliser la boucle
  parallel_for(tbb::blocked_range<int>(0, m_height), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      for (int j = 0; j < m_width; ++j) {
        double x = 1.0 * j / m_width;
        double y = 1.0 * i / m_height;

        double v = 0.0;
        for (int k = 0; k < n; k++) {
          v += charge[k].potential_at({x, y});
        }
        potentials[IDX2(i, j, m_width)] = v;

        // Mise à jour de lo et hi de manière thread-safe
        {
          tbb::spin_mutex::scoped_lock lock(mutex);  // Protège l'accès à lo et hi
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
    }
  });

  // Écrire les résultats dans m_sol après le calcul parallèle
  for (int i = 0; i < m_height; ++i) {
    for (int j = 0; j < m_width; ++j) {
      m_sol[IDX2(i, j, m_width)] = potentials[IDX2(i, j, m_width)];
    }
  }
}

void PotentialParallel::move_particles(std::vector<Particle> &charge, double dt, int substeps) {
  // TODO : Parallisation de la fonction originale dans potential.cpp
  const int n = static_cast<int>(charge.size());
  const double ssdt = dt / substeps;

  for (int ss = 0; ss < substeps; ss++) {
    // Calculer les forces entre les charges
    // Utiliser un tableau temporaire pour stocker les forces
    std::vector<Vector2d> forces(n, Vector2d::Zero());

    // Paralléliser le calcul des forces
    parallel_for(tbb::blocked_range(0, n),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                Particle &c = charge[i];
                Vector2d f = Vector2d::Zero();
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        f += c.force(charge[j]);
                    }
                }
                forces[i] = f; // Stocker la force calculée
            }
        });

    // Mettre à jour les forces dans les particules
    for (int i = 0; i < n; i++) {
      charge[i].m_f = forces[i];
    }

    // Paralléliser le déplacement des particules
    tbb::parallel_for(tbb::blocked_range(0, n),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                Particle &c = charge[i];
                // Mettre à jour la vitesse en fonction de l'accélération
                c.m_v += c.m_f * ssdt;

                // Sauvegarder la position courante
                c.m_p = c.m_x;

                // Déplacer la particule en fonction de sa vitesse
                c.m_x += c.m_v * ssdt;
            }
        });
  }
}

void PotentialParallel::save_solution(std::ostream &ofs, ColorMap &cmap) {
  // Paralléliser le traitement des pixels de l'image
  parallel_for(tbb::blocked_range(0, m_height),
      [&](const tbb::blocked_range<int>& r) {
          for (int i = r.begin(); i != r.end(); ++i) {
              for (int j = 0; j < m_width; ++j) {
              const double v = m_sol[IDX2(i, j, m_width)];
              const png::rgb_pixel pix = cmap.get_color(v);
                  // Inversion axe y pour correspondre au plan cartésien
                  m_img.set_pixel(j, m_height - i - 1, pix);
              }
          }
      });

  // Écriture de l'image dans le flux de sortie (non parallèle)
  m_img.write_stream(ofs);
}

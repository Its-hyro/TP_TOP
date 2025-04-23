#include <cassert>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>
#include <fmt/core.h>

template <typename Layout>
using Matrix = Kokkos::View<double**, Layout>;

template <class MatrixType>
auto matrix_init(MatrixType& M) -> void {
  static_assert(2 == MatrixType::rank(), "View must be of rank 2");

  Kokkos::parallel_for(
    "init",
    M.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(M.extent(1)); ++j) {
        M(i, j) = drand48();
      }
    }
  );
}

template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C) -> void {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));
  assert(A.extent(1) == B.extent(0));

  Kokkos::parallel_for(
    "dgemm_kernel",
    A.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(B.extent(1)); ++j) {
        double acc = 0.0;
        for (int k = 0; k < int(A.extent(1)); ++k) {
          acc += alpha * A(i, k) * B(k, j);
        }
        C(i, j) *= beta + acc;
      }
    }
  );
}

// Structure pour les tailles de blocs optimisées pour le cache M1
struct CacheBlockSizes {

    // L1: bloc de 32x32 doubles = 8KB (permet plusieurs blocs en L1)
    static constexpr int L1_M = 32;
    static constexpr int L1_N = 32;
    static constexpr int L1_K = 32;

    // L2: bloc de 64x64 doubles = 32KB
    static constexpr int L2_M = 64;
    static constexpr int L2_N = 64;
    static constexpr int L2_K = 64;
};

template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product_blocked(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C) -> void {
    static_assert(
        AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2,
        "Views must be of rank 2"
    );
    assert(A.extent(0) == C.extent(0));
    assert(B.extent(1) == C.extent(1));
    assert(A.extent(1) == B.extent(0));

    const int M = A.extent(0);
    const int N = B.extent(1);
    const int K = A.extent(1);

    // Parallélisation sur les blocs L2
    Kokkos::parallel_for(
        "dgemm_blocked",
        Kokkos::RangePolicy<>(0, (M + CacheBlockSizes::L2_M - 1) / CacheBlockSizes::L2_M),
        KOKKOS_LAMBDA(const int i2) {
            const int i2_start = i2 * CacheBlockSizes::L2_M;
            const int i2_end = std::min(i2_start + CacheBlockSizes::L2_M, M);

            // Parcours des blocs L2
            for (int j2 = 0; j2 < N; j2 += CacheBlockSizes::L2_N) {
                const int j2_end = std::min(j2 + CacheBlockSizes::L2_N, N);

                for (int k2 = 0; k2 < K; k2 += CacheBlockSizes::L2_K) {
                    const int k2_end = std::min(k2 + CacheBlockSizes::L2_K, K);

                    // Parcours des blocs L1
                    for (int i1 = i2_start; i1 < i2_end; i1 += CacheBlockSizes::L1_M) {
                        const int i1_end = std::min(i1 + CacheBlockSizes::L1_M, i2_end);

                        for (int j1 = j2; j1 < j2_end; j1 += CacheBlockSizes::L1_N) {
                            const int j1_end = std::min(j1 + CacheBlockSizes::L1_N, j2_end);

                            for (int k1 = k2; k1 < k2_end; k1 += CacheBlockSizes::L1_K) {
                                const int k1_end = std::min(k1 + CacheBlockSizes::L1_K, k2_end);

                                // Calcul sur les blocs L1
                                for (int i = i1; i < i1_end; ++i) {
                                    for (int j = j1; j < j1_end; ++j) {
                                        double temp = (k2 == 0 && k1 == k2) ? 
                                            (beta * C(i, j)) : C(i, j);

                                        // Déroulage optimisé pour NEON
                                        #pragma unroll 8
                                        for (int k = k1; k < k1_end; ++k) {
                                            temp += alpha * A(i, k) * B(k, j);
                                        }

                                        C(i, j) = temp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    );
}

auto main(int argc, char* argv[]) -> int {
  if (argc < 5) {
    fmt::print("Usage: {} <M> <N> <K> <layout>\n", argv[0]);
    fmt::print("layout: 'right' or 'left'\n");
    return -1;
  }

  // Configuration OpenMP optimisée
  #ifdef _OPENMP
  omp_set_num_threads(omp_get_max_threads());
  omp_set_schedule(omp_sched_dynamic, 0);
  #endif

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  std::string layout_str = argv[4];

  // Affichage des informations de configuration
  fmt::print("\nConfiguration:\n");
  fmt::print("Dimensions des matrices:\n");
  fmt::print("  A: {} x {}\n", m, k);
  fmt::print("  B: {} x {}\n", k, n);
  fmt::print("  C: {} x {}\n", m, n);
  fmt::print("Layout: {}\n", layout_str);

  // Calcul et affichage de la taille mémoire
  const size_t matrix_A_size = m * k * sizeof(double);
  const size_t matrix_B_size = k * n * sizeof(double);
  const size_t matrix_C_size = m * n * sizeof(double);
  const size_t total_size = matrix_A_size + matrix_B_size + matrix_C_size;

  fmt::print("\nUtilisation mémoire:\n");
  fmt::print("  Matrice A: {:.2f} MB\n", matrix_A_size / (1024.0 * 1024.0));
  fmt::print("  Matrice B: {:.2f} MB\n", matrix_B_size / (1024.0 * 1024.0));
  fmt::print("  Matrice C: {:.2f} MB\n", matrix_C_size / (1024.0 * 1024.0));
  fmt::print("  Total: {:.2f} MB\n", total_size / (1024.0 * 1024.0));

  // Affichage des tailles de blocs
  fmt::print("\nTailles des blocs:\n");
  fmt::print("  Cache L1: {}x{}x{}\n", 
             CacheBlockSizes::L1_M, CacheBlockSizes::L1_N, CacheBlockSizes::L1_K);
  fmt::print("  Cache L2: {}x{}x{}\n", 
             CacheBlockSizes::L2_M, CacheBlockSizes::L2_N, CacheBlockSizes::L2_K);

  // Known seed for deterministic RNG
  srand48(42);

  Kokkos::initialize(argc, argv);
  {
    if (layout_str == "right") {
      auto A = Matrix<Kokkos::LayoutRight>("A", m, k);
      auto B = Matrix<Kokkos::LayoutRight>("B", k, n);
      auto C = Matrix<Kokkos::LayoutRight>("C", m, n);

      double alpha = drand48();
      fmt::print("Initialisation des matrices...\n");
      matrix_init(A);
      matrix_init(B);
      double beta = drand48();
      matrix_init(C);

      fmt::print("Multiplication des matrices...\n");
      auto start = std::chrono::high_resolution_clock::now();
      Kokkos::fence();
      matrix_product_blocked(alpha, A, B, beta, C);
      Kokkos::fence();
      auto end = std::chrono::high_resolution_clock::now();
      
      auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
      double time_seconds = duration.count() / 1000.0;
      
      // Calcul des FLOPS
      // Pour chaque élément de C (m*n), on fait 2*k opérations (multiplication et addition)
      double flops = 2.0 * m * n * k;
      double gflops = flops / (time_seconds * 1e9);
      
      fmt::print("Temps d'exécution: {:.3f} ms\n", duration.count());
      fmt::print("Performance: {:.3f} GFLOP/s\n", gflops);
    } else if (layout_str == "left") {
      auto A = Matrix<Kokkos::LayoutLeft>("A", m, k);
      auto B = Matrix<Kokkos::LayoutLeft>("B", k, n);
      auto C = Matrix<Kokkos::LayoutLeft>("C", m, n);

      double alpha = drand48();
      fmt::print("Initialisation des matrices...\n");
      matrix_init(A);
      matrix_init(B);
      double beta = drand48();
      matrix_init(C);

      fmt::print("Multiplication des matrices...\n");
      auto start = std::chrono::high_resolution_clock::now();
      Kokkos::fence();
      matrix_product_blocked(alpha, A, B, beta, C);
      Kokkos::fence();
      auto end = std::chrono::high_resolution_clock::now();
      
      auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
      double time_seconds = duration.count() / 1000.0;
      
      // Calcul des FLOPS
      double flops = 2.0 * m * n * k;
      double gflops = flops / (time_seconds * 1e9);
      
      fmt::print("Temps d'exécution: {:.3f} ms\n", duration.count());
      fmt::print("Performance: {:.3f} GFLOP/s\n", gflops);
    } else {
      fmt::print("Erreur: layout doit être 'right' ou 'left'\n");
      return -1;
    }
  }
  Kokkos::finalize();
  return 0;
}

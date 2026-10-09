// STREAM-style RAM bandwidth test (copy / scale / add / triad), OpenMP.
// Build: gcc -O3 -march=native -fopenmp -o mem_bw mem_bw.c
// Run:   OMP_NUM_THREADS=N OMP_PROC_BIND=spread ./mem_bw [MiB per array]   (default 2048)
// Reports the best of 10 passes in GB/s (1 GB = 1e9 bytes), as STREAM does.
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? strtoul(argv[1], 0, 10) : 2048;
    size_t n = mib * 1024 * 1024 / sizeof(double);
    double *a = aligned_alloc(64, n * sizeof(double));
    double *b = aligned_alloc(64, n * sizeof(double));
    double *c = aligned_alloc(64, n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    const char *name[4] = {"copy", "scale", "add", "triad"};
    const double words[4] = {2, 2, 3, 3};
    double best[4] = {1e30, 1e30, 1e30, 1e30};
    const double s = 3.0;
    for (int pass = 0; pass < 10; pass++) {
        double t;
        t = omp_get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) c[i] = a[i];
        t = omp_get_wtime() - t; if (t < best[0]) best[0] = t;
        t = omp_get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) b[i] = s * c[i];
        t = omp_get_wtime() - t; if (t < best[1]) best[1] = t;
        t = omp_get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
        t = omp_get_wtime() - t; if (t < best[2]) best[2] = t;
        t = omp_get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
        t = omp_get_wtime() - t; if (t < best[3]) best[3] = t;
    }
    printf("threads=%d array=%zu MiB\n", omp_get_max_threads(), mib);
    for (int k = 0; k < 4; k++)
        printf("%-6s %8.1f GB/s\n", name[k], words[k] * n * sizeof(double) / best[k] / 1e9);
    return a[n / 2] < 0;  // keep the loops live
}

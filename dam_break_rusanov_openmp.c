#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define NX     10001
#define G      1.0
#define CFL    0.9
#define XLEFT -10.0
#define XRIGHT 10.0
#define XM     0.0
#define HLEFT  5.0
#define HRIGHT 2.0
#define TEND   3.0
#define MAX_THREADS 8
#define AVG_RUNS 10

void init_conditions(int nc, double *x, double *h, double *u) {
    #pragma omp parallel for
    for (int i = 0; i < nc; ++i) {
        h[i] = (x[i] < XM) ? HLEFT : HRIGHT;
        u[i] = 0.0;
    }
}

void rusanov_flux(int nc, double *h, double *u, double flux[2][NX - 1]) {
    #pragma omp parallel for
    for (int i = 0; i < nc - 1; ++i) {
        double hL = h[i], hR = h[i + 1];
        double uL = u[i], uR = u[i + 1];

        double WL[2] = {hL, hL * uL};
        double WR[2] = {hR, hR * uR};

        double FL_L[2] = {hL * uL, hL * uL * uL + 0.5 * G * hL * hL};
        double FL_R[2] = {hR * uR, hR * uR * uR + 0.5 * G * hR * hR};

        double sL = fabs(uL) + sqrt(G * hL);
        double sR = fabs(uR) + sqrt(G * hR);
        double smax = fmax(sL, sR);

        for (int j = 0; j < 2; ++j)
            flux[j][i] = 0.5 * (FL_L[j] + FL_R[j]) - 0.5 * smax * (WR[j] - WL[j]);
    }
}

int main() {
    double T_seq = 0.0;
    double results[MAX_THREADS][3] = {0}; // [time, speedup, efficiency]

    for (int threads = 1; threads <= MAX_THREADS; threads++) {
        double total_time = 0.0;

        for (int run = 0; run < AVG_RUNS; run++) {
            omp_set_num_threads(threads);

            int nx = NX;
            int nc = nx - 1;
            double dx = (XRIGHT - XLEFT) / (nx - 1);

            double x[NX], xc[NX - 1];
            double h[NX - 1], u[NX - 1], hn[NX - 1], un[NX - 1];
            double W[2][NX - 1], Wn[2][NX - 1], flux[2][NX - 1];

            clock_t start_time = clock();

            #pragma omp parallel for
            for (int i = 0; i < nx; ++i)
                x[i] = XLEFT + i * dx;

            #pragma omp parallel for
            for (int i = 0; i < nc; ++i)
                xc[i] = 0.5 * (x[i] + x[i + 1]);

            init_conditions(nc, x, h, u);

            #pragma omp parallel for
            for (int i = 0; i < nc; ++i) {
                W[0][i] = h[i];
                W[1][i] = h[i] * u[i];
            }

            double t = 0.0;
            while (t < TEND) {
                double max_speed = 0.0;

                #pragma omp parallel for reduction(max:max_speed)
                for (int i = 0; i < nc; ++i) {
                    h[i] = W[0][i];
                    u[i] = W[1][i] / (h[i] > 1e-6 ? h[i] : 1e-6);
                    double c = sqrt(G * h[i]);
                    double speed = fabs(u[i]) + c;
                    if (speed > max_speed)
                        max_speed = speed;
                }

                double dt = CFL * dx / max_speed;
                if (t + dt > TEND)
                    dt = TEND - t;
                double nu = dt / dx;
                t += dt;

                rusanov_flux(nc, h, u, flux);

                #pragma omp parallel for
                for (int i = 1; i < nc - 1; ++i) {
                    for (int j = 0; j < 2; ++j)
                        Wn[j][i] = W[j][i] - nu * (flux[j][i] - flux[j][i - 1]);
                }

                for (int j = 0; j < 2; ++j) {
                    Wn[j][0] = Wn[j][1];
                    Wn[j][nc - 1] = Wn[j][nc - 2];
                }

                #pragma omp parallel for
                for (int i = 0; i < nc; ++i) {
                    W[0][i] = Wn[0][i];
                    W[1][i] = Wn[1][i];
                }
            }

            clock_t end_time = clock();
            total_time += ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

            if (threads == 8 && run == AVG_RUNS - 1) {
                FILE *f = fopen("output_openmp.dat", "w");
                for (int i = 0; i < nc; ++i)
                    fprintf(f, "%f\t%f\t%f\n", xc[i], W[0][i], W[1][i] / W[0][i]);
                fclose(f);
            }
        }

        double avg_time = total_time / AVG_RUNS;
        if (threads == 1)
            T_seq = avg_time;

        double speedup = T_seq / avg_time;
        double efficiency = speedup / threads;

        results[threads - 1][0] = avg_time;
        results[threads - 1][1] = speedup;
        results[threads - 1][2] = efficiency;
    }

    printf("\n==== Résumé des performances moyennes (%d exécutions) ====\n", AVG_RUNS);
    printf("Threads\tTemps(s)\tSpeedup\t\tEfficacité(%%)\n");
    for (int i = 0; i < MAX_THREADS; ++i) {
        printf("%d\t%.6f\t%.2f\t\t%.2f%%\n",
               i + 1,
               results[i][0],
               results[i][1],
               results[i][2] * 100.0);
    }

    return 0;
}

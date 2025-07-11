#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NX     10001
#define G      1.0
#define CFL    0.9
#define XLEFT -10.0
#define XRIGHT 10.0
#define XM     0.0
#define HLEFT  5.0
#define HRIGHT 2.0
#define TEND   3.0

double min(double a, double b) { return a < b ? a : b; }
double max(double a, double b) { return a > b ? a : b; }

// Rusanov flux (1D)
void rusanov_flux(int nc, double *h, double *u, double *flux) {
    for (int i = 0; i <= nc; ++i) {
        double hL = h[i], hR = h[i + 1];
        double uL = u[i], uR = u[i + 1];
        double qL = hL * uL;
        double qR = hR * uR;

        double FL[2] = { qL, qL * uL + 0.5 * G * hL * hL };
        double FR[2] = { qR, qR * uR + 0.5 * G * hR * hR };

        double smax = fmax(fabs(uL) + sqrt(G * hL), fabs(uR) + sqrt(G * hR));
        flux[0 * (nc + 1) + i] = 0.5 * (FL[0] + FR[0]) - 0.5 * smax * (hR - hL);
        flux[1 * (nc + 1) + i] = 0.5 * (FL[1] + FR[1]) - 0.5 * smax * (qR - qL);
    }
}

// Initial conditions
void init_conditions(int nc_local, int start_index, double dx, double *x, double *h, double *u) {
    for (int i = 1; i <= nc_local; ++i) {
        x[i] = XLEFT + (start_index + i - 1 + 0.5) * dx;//we  don’t store the full x we must calculate only our local centers
        h[i] = (x[i] < XM) ? HLEFT : HRIGHT;
        u[i] = 0.0;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nc_global = NX - 1;
    int base = nc_global / size;
    int rest = nc_global % size;
    int nc_local = base + (rank < rest ? 1 : 0);
    int start_index = rank * base + min(rank, rest);

    double dx = (XRIGHT - XLEFT) / (NX - 1);

    // Allocation with ghost cells
    double *x = malloc((nc_local + 2) * sizeof(double));
    double *h = malloc((nc_local + 2) * sizeof(double));
    double *u = malloc((nc_local + 2) * sizeof(double));
    double *hn = malloc((nc_local + 2) * sizeof(double));
    double *un = malloc((nc_local + 2) * sizeof(double));
    double *flux = malloc(2 * (nc_local + 1) * sizeof(double));

    // Initialize conditions
    init_conditions(nc_local, start_index, dx, x, h, u);

    double t = 0.0;
    while (t < TEND) {
        h[0] = h[1]; h[nc_local + 1] = h[nc_local];
        u[0] = u[1]; u[nc_local + 1] = u[nc_local];

        if (rank > 0) {
            MPI_Sendrecv(&h[1], 1, MPI_DOUBLE, rank - 1, 0,
                         &h[0], 1, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, rank - 1, 2,
                         &u[0], 1, MPI_DOUBLE, rank - 1, 3,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&h[nc_local], 1, MPI_DOUBLE, rank + 1, 1,
                         &h[nc_local + 1], 1, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&u[nc_local], 1, MPI_DOUBLE, rank + 1, 3,
                         &u[nc_local + 1], 1, MPI_DOUBLE, rank + 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double local_max = 0.0;
        for (int i = 1; i <= nc_local; ++i) {
            double c = sqrt(G * h[i]);
            double v = fabs(u[i]) + c;
            if (v > local_max) local_max = v;
        }

        double global_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double dt = CFL * dx / global_max;
        if (t + dt > TEND) dt = TEND - t;
        double nu = dt / dx;
        t += dt;

        rusanov_flux(nc_local, h, u, flux);

        for (int i = 1; i <= nc_local; ++i) {
            hn[i] = h[i] - nu * (flux[0 * (nc_local + 1) + i] - flux[0 * (nc_local + 1) + i - 1]);
            un[i] = (h[i] > 1e-6) ?
                (h[i] * u[i] - nu * (flux[1 * (nc_local + 1) + i] - flux[1 * (nc_local + 1) + i - 1])) / hn[i] :
                0.0;
        }

        for (int i = 1; i <= nc_local; ++i) {
            h[i] = hn[i];
            u[i] = un[i];
        }

        if (rank == 0)
            printf("t = %.4f\n", t);
    }

    // Gather local data
    double *local_x = malloc(nc_local * sizeof(double));
    double *local_h = malloc(nc_local * sizeof(double));
    double *local_u = malloc(nc_local * sizeof(double));
    for (int i = 0; i < nc_local; ++i) {
        local_x[i] = x[i + 1];
        local_h[i] = h[i + 1];
        local_u[i] = u[i + 1];
    }

    int *recvcounts = NULL, *displs = NULL;
    double *gather_x = NULL, *gather_h = NULL, *gather_u = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = base + (i < rest ? 1 : 0);
            displs[i] = offset;
            offset += recvcounts[i];
        }
        gather_x = malloc(nc_global * sizeof(double));
        gather_h = malloc(nc_global * sizeof(double));
        gather_u = malloc(nc_global * sizeof(double));
    }

    MPI_Gatherv(local_x, nc_local, MPI_DOUBLE, gather_x, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_h, nc_local, MPI_DOUBLE, gather_h, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_u, nc_local, MPI_DOUBLE, gather_u, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *f = fopen("output_mpi.dat", "w");
        for (int i = 0; i < nc_global; ++i)
            fprintf(f, "%f\t%f\t%f\n", gather_x[i], gather_h[i], gather_u[i]);
        fclose(f);
        printf("Simulation MPI terminée. Résultats écrits dans output_mpi.dat\n");
        free(gather_x); free(gather_h); free(gather_u);
        free(recvcounts); free(displs);
    }

    free(x); free(h); free(u); free(hn); free(un); free(flux);
    free(local_x); free(local_h); free(local_u);

    MPI_Finalize();
    return 0;
}
